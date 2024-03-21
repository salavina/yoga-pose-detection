from ypd.config.configuration import configurationManager
from ypd.components.model_training import ModelTrainer, ModelTrainerViT
from ypd import logger




STAGE_NAME = "Model Training"


class ModelTrainingPipeline:
    def __init__(self) -> None:
        pass
    
    def main(self):
        config = configurationManager()
        training_config = config.get_training_config()
        training = ModelTrainer(config=training_config)
        training.train()
        # turn on & off mlflow tracking here
        training.log_into_mlflow()

class ModelTrainingPipelineViT:
    def __init__(self) -> None:
        pass
    
    def main(self):
        config = configurationManager()
        training_config = config.get_training_config()
        training = ModelTrainerViT(config=training_config)
        training.train()
        # turn on & off mlflow tracking here
        # training.log_into_mlflow()




if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipelineViT()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e