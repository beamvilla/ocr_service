from utils import load_yaml_file


class TrainConfig:
    def __init__(self, config_path: str = "./config/train.yaml") -> None:
        config_file = load_yaml_file(config_path)
        self.EPOCHS: int = config_file["EPOCHS"]
        self.DEVICE: str = config_file["DEVICE"]
        self.MODEL_DIR: str = config_file["MODEL_DIR"]
        self.BATCH_SIZE: int = config_file["BATCH_SIZE"]
        self.PATIENCE: int = config_file["PATIENCE"]
        self.INPUT_WIDTH: int = config_file["INPUT_WIDTH"]
        self.INPUT_HEIGHT: int = config_file["INPUT_HEIGHT"]