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