
from ypd.config.configuration import configurationManager
from ypd.components.data_ingestion import DataIngestion
from ypd import logger


STAGE_NAME = "Data Ingestion Stage"


class DataIngestionTrainingPipeline():
    def __init__(self) -> None:
        pass
    
    def main(self):
        config = configurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        



if __name__ == "__main__":
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<<<\nx===============x")
    
    except Exception as e:
        logger.exception(e)
        raise e