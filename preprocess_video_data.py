import argparse
import json
import os
from functools import partial
from multiprocessing import Pool, current_process

import cv2
import numpy as np
from tqdm import tqdm

os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
cv2.setNumThreads(0)


def init_worker():
    pid = current_process().pid
    print(f"Worker {pid} initialized.")


def process_single_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # return rgb frames [T, H, W, 3]
    return rgb_frame.astype(np.float32) / 255.0


def process_video(video_id, config):
    video_path = os.path.join(config["video_dir"], f"{video_id}.mp4")
    output_path = os.path.join(config["output_dir"], f"{video_id}.npz")

    try:
        if config["skip_existing"] and os.path.exists(output_path):
            return {"video_id": video_id, "status": "skipped"}

        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(process_single_frame(frame))

        frames = np.array(frames)
        if frames.shape[0] == 0:
            raise ValueError("No frames found in video")

        np.savez_compressed(output_path, frames=frames)

        return {"video_id": video_id, "status": "success"}
    except Exception as e:
        return {
            "video_id": video_id,
            "status": "error",
            "message": str(e),
        }
    finally:
        if "cap" in locals():
            cap.release()  # type: ignore


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
        default="data/processed/videos",
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

    if not os.path.exists(config["output_dir"]):
        os.makedirs(config["output_dir"])

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
