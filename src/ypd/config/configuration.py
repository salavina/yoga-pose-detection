from ypd.constants import *
from ypd.utils.common import read_yaml, create_directories
from ypd.entity.config_entity import (DataIngestionConfig,
                                      PrepareBaseModelConfig,
                                      TrainingConfig)
import os
from dotenv import load_dotenv

load_dotenv()

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
MLFLOW_TRACKING_USERNAME = os.environ["MLFLOW_TRACKING_USERNAME"]
MLFLOW_TRACKING_PASSWORD = os.environ["MLFLOW_TRACKING_PASSWORD"]

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
            vit_feature_extractor = config.vit_feature_extractor,
            vit_config = config.vit_config,
            vit_updated_base_model_path = config.vit_updated_base_model_path,
            params_classes = self.params.CLASSES,
            params_image_size = self.params.IMAGE_SIZE,
            params_pretrained = self.params.PRETRAINED,
            params_type = self.params.TYPE
        )
        
        return prepare_base_model_config
    
    
    def get_training_config(self) -> TrainingConfig:
        model_training = self.config.model_training
        prepare_base_model = self.config.prepare_base_model
        training_data = os.path.join(self.config.data_ingestion.root_dir, 'yoga-poses-dataset')
        
        create_directories([model_training.root_dir])
        
        training_config = TrainingConfig(
            root_dir= model_training.root_dir,
            resnet_trained_model_path= model_training.resnet_trained_model_path,
            resnet_updated_base_model_path= prepare_base_model.resnet_updated_base_model_path,
            vit_trained_model_path= model_training.vit_trained_model_path,
            training_data= training_data,
            mlflow_uri = MLFLOW_TRACKING_URI,
            all_params = self.params,
            params_augmentation= self.params.AUGMENTATION,
            params_image_size= self.params.IMAGE_SIZE,
            params_batch_size= self.params.BATCH_SIZE,
            params_epochs= self.params.EPOCHS,
            params_learning_rate= self.params.LEARNING_RATE
        )
        
        return training_config