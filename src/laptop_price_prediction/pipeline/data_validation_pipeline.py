from src.laptop_price_prediction import logger
from src.laptop_price_prediction.config.configuration import ConfigurationManager
from src.laptop_price_prediction.components.data_validation import DataValidation
from src.laptop_price_prediction.entity.config_entity import (DataValidationConfig)
from src.laptop_price_prediction.constants import *


STAGE_NAME = "Data Validation Stage"

class DataValidationPipeline:
    def __init__(self):
        pass
    def InitiateDataValidationPipeline(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        data_validation.validate_data()

if __name__=="__main__":
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
        obj=DataValidationPipeline()
        obj.InitiateDataValidationPipeline()
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

