from src.laptop_price_prediction.config.configuration import ConfigurationManager
from src.laptop_price_prediction.components.data_ingestion import DataIngestion
from src.laptop_price_prediction import logger


STAGE_NAME="Data Ingestion Stage"


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass
    def InitiateDataIngestionTrainingPipeline(self):
            config=ConfigurationManager()
            data_ingestion_config=config.get_data_ingestion_config()
            data_ingestion=DataIngestion(config=data_ingestion_config)
            data_ingestion.download_file()
            data_ingestion.extract_zip_file()
            logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx==========x")
    
if __name__=="__main__":
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
        obj=DataIngestionTrainingPipeline()
        obj.InitiateDataIngestionTrainingPipeline()
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e