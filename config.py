import yaml


class TrainConfig:
    def __init__(self, config_path: str):
        config = yaml.safe_load(open(config_path, "r"))

        # Data
        data_config = config.get("data", {})
        self.data_root_path = data_config.get("path", "data/raw")
        self.train_split = data_config.get("train_split", "nslt_2000.json")

        # Train
        train_config = config.get("train", {})
        ...

    def __str__(self):
        return f"TrainConfig(data_root_path={self.data_root_path}, train_split={self.train_split})"


if __name__ == "__main__":
    config = TrainConfig("configs/default.yaml")
    print(config)
