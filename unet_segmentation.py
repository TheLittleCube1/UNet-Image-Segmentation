import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

loaded_unet = tf.keras.models.load_model("unet.h5")

COLORS_BGR = np.array([
    [180, 126, 123],    # 0 Unlabeled
    [0, 75, 150],       # 1 Building
    [40, 40, 90],       # 2 Fence
    [180, 126, 123],    # 3 Other
    [0, 140, 255],      # 4 Pedestrian
    [128, 128, 128],    # 5 Pole
    [255, 255, 255],    # 6 Road Line
    [0, 0, 0],          # 7 Road
    [200, 200, 200],    # 8 Sidewalk
    [114, 208, 114],    # 9 Vegetation
    [0, 0, 255],        # 10 Vehicle
    [90, 90, 90],       # 11 Wall
    [0, 140, 255],      # 12 Traffic Sign
    [230, 216, 173],    # 13 Sky
    [0, 0, 0],          # 14 Ground
    [120, 80, 40],      # 15 Bridge
    [80, 80, 120],      # 16 Rail Track
    [0, 255, 0],        # 17 Guard Rail
    [0, 0, 190],        # 18 Traffic Light
    [0, 0, 255],        # 19 Static
    [0, 0, 255],        # 20 Dynamic
    [255, 100, 0],      # 21 Water
    [80, 200, 80],      # 22 Terrain
], dtype=np.uint8)

CARLA_CLASSES = ["Unlabeled", "Building", "Fence", "Other", "Pedestrian", "Pole", "Road Line", "Road", "Sidewalk", "Vegetation", "Vehicle", "Wall", "Traffic Sign", "Sky", "Road", "Bridge", "Rail Track", "Guard Rail", "Traffic Light", "Static", "Dynamic", "Water", "Terrain"]

LEGEND_CLASSES = [0, 1, 4, 8, 6, 14, 9, 10]

def timestamp_to_seconds(ts):
    parts = [int(p) for p in ts.split(":")]
    if len(parts) == 2:
        m, s = parts
        return m * 60 + s
    elif len(parts) == 3:
        h, m, s = parts
        return h * 3600 + m * 60 + s
    else:
        raise ValueError("Invalid timestamp format")

def seconds_to_frame(seconds, fps):
    return int(seconds * fps)

def crop_center(frame):
    h, w = frame.shape[:2]
    target_ratio = 4.0 / 3.0
    if w / h > target_ratio:
        new_w = int(h * target_ratio)
        x1 = (w - new_w) // 2
        return frame[:, x1:x1 + new_w]
    else:
        new_h = int(w / target_ratio)
        y1 = (h - new_h) // 2
        return frame[y1:y1 + new_h, :]

def preprocess(frame, width=256, height=256):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (width, height))
    frame = frame.astype(np.float32) / 255.0
    return np.expand_dims(frame, axis=0)

def predict_mask(model, frame):
    x = preprocess(frame, width=256, height=256)
    pred = model.predict(x, verbose=0)
    return np.argmax(pred[0], axis=-1)

def colorize_mask(mask):
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(len(COLORS_BGR)):
        out[mask == i] = COLORS_BGR[i]
    return out

def draw_legend(frame, colors):
    overlay = frame.copy()
    x0, y0 = 10, 10
    box_h = 22
    box_w = 20
    cv2.rectangle(
        overlay,
        (x0, y0),
        (x0 + 130, y0 + len(LEGEND_CLASSES) * box_h + 10),
        (0, 0, 0),
        -1
    )

    frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)

    for abridged_i in range(len(LEGEND_CLASSES)):
        i = LEGEND_CLASSES[abridged_i]
        name = CARLA_CLASSES[i]
        y = y0 + 20 + abridged_i * box_h
        color = tuple(int(c) for c in colors[i])
        cv2.rectangle(frame, (x0 + 5, y - 12), (x0 + 25, y + 5), color, -1)
        cv2.putText(
            frame,
            f"{name}",
            (x0 + 35, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
    return frame

def overlay_mask(frame, mask):
    mask_color = colorize_mask(mask)
    mask_color = cv2.resize(
        mask_color,
        (frame.shape[1], frame.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )
    out = cv2.addWeighted(frame, 0.6, mask_color, 0.4, 0)
    return draw_legend(out, COLORS_BGR)

def run_video(
    model=loaded_unet,
    intervals=None,
    input_path="Toronto Driving Footage.mp4",
    output_path="Masked Output.mp4",
    frame_skip=3
):
    if intervals is None:
        intervals = []
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: cannot open video")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    writer = None
    for start_ts, end_ts in intervals:
        start_frame = seconds_to_frame(timestamp_to_seconds(start_ts), fps)
        end_frame = seconds_to_frame(timestamp_to_seconds(end_ts), fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_idx = start_frame
        last_mask = None
        while frame_idx <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center(frame)
            crop_bottom = 110
            if writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h - crop_bottom))
            if last_mask is None or (frame_idx - start_frame) % frame_skip == 0:
                last_mask = predict_mask(model, frame)
            output = overlay_mask(frame, last_mask)
            cv2.imshow("Image Segmentation", output[:-crop_bottom, :])
            writer.write(output[:-crop_bottom, :])
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cap.release()
                if writer:
                    writer.release()
                cv2.destroyAllWindows()
                return
            frame_idx += 1
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

intervals = [
    ("56:45", "60:45"),
    ("43:20", "43:50"),
    ("31:40", "34:00"),
    ("28:30", "28:50"),
    ("22:05", "23:35"),
    # ("1:00", "5:00")
]

run_video(model=loaded_unet, intervals=intervals, frame_skip=3)
# crop_video()