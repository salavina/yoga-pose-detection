from ypd.constants import *
from ypd.utils.common import read_yaml, create_directories
from ypd.entity.config_entity import (DataIngestionConfig,
                                      PrepareBaseModelConfig)

class configurationManager:
    def __init__(self, config_file_path = CONFIG_FILE_PATH,
                 params_file_path = PARAMS_FILE_PATH):
        self.config = read_yaml(config_file_path)
        self.params = read_yaml(params_file_path)
        
        create_directories([self.config.artifacts_root])
        
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        
        create_directories([config.root_dir])
        
        data_ingestion_config = DataIngestionConfig(
            root_dir = config.root_dir,
            source_URL = config.source_URL
        )
        
        return data_ingestion_config
    

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        
        create_directories([config.root_dir])
        
        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir = config.root_dir,
            resnet_base_model_path = config.resnet_base_model_path,
            resnet_updated_base_model_path = config.resnet_updated_base_model_path,
            params_classes = self.params.CLASSES,
            params_image_size = self.params.IMAGE_SIZE,
            params_pretrained = self.params.PRETRAINED
        )
        
        return prepare_base_model_config