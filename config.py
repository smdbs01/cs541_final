import yaml


class TrainConfig:
    def __init__(self, config_path: str):
        config = yaml.safe_load(open(config_path, "r"))

        # Data
        data_config = config.get("data", {})
        self.data_root: str = data_config.get("path", "data/raw")
        self.train_split: str = data_config.get("train_split", "nslt_2000.json")
        self.mode: str = data_config.get("mode", "rgb")

        # Train
        train_config = config.get("train", {})
        self.batch_size: int = train_config.get("batch_size", 32)

    def __str__(self):
        return (
            f"TrainConfig("
            f"data_root={self.data_root}, mode={self.mode}, train_split={self.train_split},"
            f")"
        )


if __name__ == "__main__":
    config = TrainConfig("configs/default.yaml")
    print(config)
