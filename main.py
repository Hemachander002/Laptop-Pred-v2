from src.laptop_price_prediction.pipeline.data_validation_pipeline import DataValidationPipeline
from src.laptop_price_prediction import logger
from src.laptop_price_prediction.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.laptop_price_prediction.pipeline.data_transformation_pipeline import DataTransformationPipeline

STAGE_NAME="Data Ingestion Stage"
try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
    obj=DataIngestionTrainingPipeline()
    obj.InitiateDataIngestionTrainingPipeline()
    logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME="Data Validation Stage"
try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
    obj=DataValidationPipeline()
    obj.InitiateDataValidationPipeline()
    logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Transformation Stage"
if __name__=="__main__":
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
        obj=DataTransformationPipeline()
        obj.InitiateDataTransformationPipeline()
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

