import json
import os

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

mp_holistic = mp.solutions.holistic

LANDMARK_COUNTS = {"face": 468, "pose": 33, "left_hand": 21, "right_hand": 21}


def process_single_frame(frame, holistic):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb_frame)

    frame_data = {}

    # process landmarks for each body part
    def process_landmarks(landmarks, fallback_count):
        if landmarks:
            count = len(landmarks.landmark)
            arr = np.zeros((count, 3), dtype=np.float32)
            for i, lm in enumerate(landmarks.landmark):
                arr[i] = [lm.x, lm.y, lm.z]
            return arr
        return np.zeros((fallback_count, 3), dtype=np.float32)

    frame_data["face"] = process_landmarks(
        results.face_landmarks,
        fallback_count=468,
    )

    frame_data["pose"] = process_landmarks(results.pose_landmarks, fallback_count=33)

    frame_data["left_hand"] = process_landmarks(
        results.left_hand_landmarks, fallback_count=21
    )

    frame_data["right_hand"] = process_landmarks(
        results.right_hand_landmarks, fallback_count=21
    )

    return frame_data


def process_video(video_path, holistic):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(process_single_frame(frame, holistic))

    cap.release()

    video_data = {
        "face": np.array([f["face"] for f in frames]),
        "pose": np.array([f["pose"] for f in frames]),
        "left_hand": np.array([f["left_hand"] for f in frames]),
        "right_hand": np.array([f["right_hand"] for f in frames]),
    }

    return video_data


def main():
    BASE_DIR = "data/raw/wlasl-complete"
    if not os.path.exists(BASE_DIR):
        raise FileNotFoundError(f"Directory {BASE_DIR} does not exist.")

    SPLIT_FILE_PATH = os.path.join(BASE_DIR, "nslt_2000.json")
    VIDEO_DIR = os.path.join(BASE_DIR, "videos")
    OUTPUT_DIR = "data/processed/landmarks"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(SPLIT_FILE_PATH) as f:
        split_data = json.load(f)

    video_ids = list(split_data.keys())

    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        refine_face_landmarks=True,
    )

    for vid in tqdm(video_ids, desc="Processing videos"):
        video_path = os.path.join(VIDEO_DIR, f"{vid}.mp4")
        if not os.path.exists(video_path):
            print(f"Video {video_path} does not exist. Skipping.")
            continue

        try:
            data = process_video(video_path, holistic)
            output_path = os.path.join(OUTPUT_DIR, f"{vid}.npz")
            np.savez(output_path, **data)
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            continue

    holistic.close()
    print("Processing complete.")

    # test
    files = os.listdir(OUTPUT_DIR)
    for file in files:
        data = np.load(os.path.join(OUTPUT_DIR, file), allow_pickle=True)
        print(data.keys())
        for k, v in data.items():
            print(k, v.shape)
        break
    print(len(files))


if __name__ == "__main__":
    main()
