from dataclasses import dataclass
from pathlib import Path

# specifies the type of value related to the key in yaml file
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    resnet_base_model_path: Path
    resnet_updated_base_model_path: Path
    params_classes: int
    params_image_size: list
    params_pretrained: bool
    

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    resnet_trained_model_path: Path
    resnet_updated_base_model_path: Path
    training_data: Path
    mlflow_uri: str
    all_params: dict
    params_augmentation: bool
    params_image_size: list
    params_batch_size: int
    params_epochs: int
    params_learning_rate: float