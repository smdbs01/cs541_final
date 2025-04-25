import argparse
import json
import os
from functools import partial
from multiprocessing import Pool, current_process

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

mp_holistic = mp.solutions.holistic
_holistic_instances = {}

os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
cv2.setNumThreads(0)


def init_worker():
    pid = current_process().pid
    _holistic_instances[pid] = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        refine_face_landmarks=True,
    )
    print(f"Worker {pid} initialized.")


def get_holistic():
    return _holistic_instances[current_process().pid]


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

    frame_data["face"] = process_landmarks(results.face_landmarks, fallback_count=478)

    frame_data["pose"] = process_landmarks(results.pose_landmarks, fallback_count=33)

    frame_data["left_hand"] = process_landmarks(
        results.left_hand_landmarks, fallback_count=21
    )

    frame_data["right_hand"] = process_landmarks(
        results.right_hand_landmarks, fallback_count=21
    )

    return frame_data


def process_video(video_id, config):
    video_path = os.path.join(config["video_dir"], f"{video_id}.mp4")
    output_path = os.path.join(config["output_dir"], f"{video_id}.npz")

    try:
        if config["skip_existing"] and os.path.exists(output_path):
            return {"video_id": video_id, "status": "skipped"}

        holistic = get_holistic()
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(process_single_frame(frame, holistic))

        video_data = {
            "face": np.array([f["face"] for f in frames]),
            "pose": np.array([f["pose"] for f in frames]),
            "left_hand": np.array([f["left_hand"] for f in frames]),
            "right_hand": np.array([f["right_hand"] for f in frames]),
        }

        np.savez_compressed(output_path, **video_data)
        return {"video_id": video_id, "status": "success"}
    except Exception as e:
        return {
            "video_id": video_id,
            "status": "error",
            "message": str(e),
        }
    finally:
        if "cap" in locals():
            cap.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        type=str,
        default="data/raw/wlasl-complete/videos",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/processed/landmarks",
    )
    parser.add_argument(
        "--split_file",
        type=str,
        default="data/raw/wlasl-complete/nslt_2000.json",
    )
    parser.add_argument(
        "-n",
        "--num_workers",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
    )
    args = parser.parse_args()
    config = {
        "video_dir": args.video_path,
        "output_dir": args.output_path,
        "skip_existing": args.skip_existing,
    }

    with open(args.split_file) as f:
        video_ids = list(json.load(f).keys())

    with Pool(args.num_workers, initializer=init_worker, initargs=()) as pool:
        worker = partial(process_video, config=config)

        results = []
        with tqdm(total=len(video_ids), desc="Processing videos") as pbar:
            for result in pool.imap_unordered(worker, video_ids):
                results.append(result)
                pbar.update(1)
                if result["status"] == "error":
                    pbar.write(f"Error {result['video_id']}: {result['message'][:50]}")


if __name__ == "__main__":
    main()
