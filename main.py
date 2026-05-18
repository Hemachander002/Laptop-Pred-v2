from src.laptop_price_prediction.pipeline.data_validation_pipeline import DataValidationPipeline
from src.laptop_price_prediction import logger
from src.laptop_price_prediction.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline

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