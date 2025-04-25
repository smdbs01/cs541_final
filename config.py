import yaml


class TrainConfig:
    def __init__(self, config_path: str):
        config = yaml.safe_load(open(config_path, "r"))

        # Data
        data_config = config.get("data", {})
        self.data_root: str = data_config.get("root", "data/raw/wlasl-complete")
        self.train_split: str = data_config.get("train_split", "nslt_2000.json")
        self.class_list: str = data_config.get(
            "class_list", "data/raw/wlasl-complete/wlasl_class_list.txt"
        )

        self.mode = data_config.get("mode", "vision")

        self.video_path: str = data_config.get(
            "video_path", "data/raw/wlasl-complete/videos"
        )
        self.pose_path: str = data_config.get("pose_path", "data/processed/landmarks")

        # Train
        train_config = config.get("train", {})
        self.batch_size: int = train_config.get("batch_size", 32)
        # Below are only for Pose
        self.n_samples: int = train_config.get("n_samples", 50)
        self.num_copies: int = train_config.get("num_copies", 4)

    def __str__(self):
        return (
            f"TrainConfig("
            f"data_root={self.data_root}, "
            f"train_split={self.train_split}, "
            f"class_list={self.class_list}, "
            f"mode={self.mode}, "
            f"video_path={self.video_path}, "
            f"pose_path={self.pose_path}, "
            f"batch_size={self.batch_size}, "
            f"n_samples={self.n_samples}, "
            f"num_copies={self.num_copies}"
            f")"
        )


if __name__ == "__main__":
    config = TrainConfig("configs/default.yaml")
    print(config)
