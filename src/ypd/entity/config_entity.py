from dataclasses import dataclass
from pathlib import Path

# specifies the type of value related to the key in yaml file
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str

