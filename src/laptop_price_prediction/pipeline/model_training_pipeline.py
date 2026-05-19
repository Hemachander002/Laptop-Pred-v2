from src.laptop_price_prediction import logger
from src.laptop_price_prediction.config.configuration import ConfigurationManager
from src.laptop_price_prediction.components.model_trainer import ModelTrainer
from src.laptop_price_prediction.entity.config_entity import (ModelTrainerConfig)
from src.laptop_price_prediction.constants import *
from pathlib import Path
import os   

STAGE_NAME = "Model Trainer Stage"

class ModelTrainerPipeline:
    def __init__(self):
        pass
    def InitiateModelTrainerPipeline(self):
        try:
            config = ConfigurationManager()
            model_trainer_config = config.get_model_trainer_config()
            model_trainer = ModelTrainer(config=model_trainer_config)
            model_trainer.train_model()
       
        except Exception as e:
            logger.exception(e)
            raise e


if __name__=="__main__":
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
        obj=ModelTrainerPipeline()
        obj.InitiateModelTrainerPipeline()
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e