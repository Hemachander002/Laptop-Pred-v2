from src.laptop_price_prediction import logger
from src.laptop_price_prediction.config.configuration import ConfigurationManager
from src.laptop_price_prediction.components.data_transformation import DataTransformation
from src.laptop_price_prediction.entity.config_entity import (DataTransformationConfig)
from src.laptop_price_prediction.constants import *
from pathlib import Path
import os   

STAGE_NAME = "Data Transformation Stage"

class DataTransformationPipeline:
    def __init__(self):
        pass
    def InitiateDataTransformationPipeline(self):
        try:
            with open(Path("artifacts/data_validation/status.txt"), "r") as f:
                status = f.read().split(" ")[-1]
                status = status.strip()
            if status == "True":
                config = ConfigurationManager()
                data_transformation_config = config.get_data_transformation_config()
                data_transformation = DataTransformation(config=data_transformation_config)
                data_transformation.transform_data()
                transformed_data = data_transformation.transform_data()
                data_transformation.split_data(transformed_data)
            else:
                print(type(status))
                print(status)
                raise Exception("Data validation failed. Data transformation cannot proceed.")
                logger.info("Data validation failed. Data transformation cannot proceed.")
        except Exception as e:
            logger.exception(e)
            raise e



if __name__=="__main__":
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
        obj=DataTransformationPipeline()
        obj.InitiateDataTransformationPipeline()
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e