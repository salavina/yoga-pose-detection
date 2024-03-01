import opendatasets as od
from ypd import logger
from ypd.utils.common import get_size
import os
import shutil
from ypd.entity.config_entity import (DataIngestionConfig)

class DataIngestion():
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    
    def download_file(self) -> str:
        
        try:
            file_name = 'kaggle.json'
            dataset_url = self.config.source_URL
            # Check if the folder path exists
            files_in_folder = os.listdir(self.config.root_dir)
            
            # Check if the specified file exists in the folder
            if file_name not in files_in_folder:
                shutil.copy(os.path.join('research/', file_name), self.config.root_dir)
            logger.info(f"Downloading data from {dataset_url} to {str(self.config.root_dir)}")
            os.chdir(self.config.root_dir)
            od.download(dataset_url)
            logger.info(f"Downloaded data from {dataset_url} to {str(self.config.root_dir)}")
        
        except Exception as e:
            raise e


