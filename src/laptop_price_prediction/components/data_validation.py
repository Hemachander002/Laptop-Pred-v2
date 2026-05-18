import os
from src.laptop_price_prediction import logger
from src.laptop_price_prediction.entity.config_entity import (DataValidationConfig)
import pandas as pd


class DataValidation:
    def __init__(self,config:DataValidationConfig):
        self.config=config
    
    def validate_data(self) -> bool:
        try:
            column_status = False
            dtype_status = False
            validation_status = False

            data = pd.read_csv(self.config.unzip_data_dir)
            expected_columns = set(self.config.all_schema.keys())
            actual_columns = set(data.columns)
            if expected_columns != actual_columns:
                logger.error(f"Data validation failed! Expected columns: {expected_columns}, Actual columns: {actual_columns}")
                column_status = False
            else:
                column_status = True
                logger.info("Column validation successful! All expected columns are present.")
                for column, expected_dtype in self.config.all_schema.items():
                    actual_dtype = data[column].dtype
                    if actual_dtype != expected_dtype:
                        logger.error(f"Data validation failed! Column '{column}' has dtype '{actual_dtype}', expected '{expected_dtype}'")
                        dtype_status = False
                        break
                    else:
                        dtype_status = True
            if dtype_status == True and column_status == True:
                validation_status = True
                with open(self.config.STATUS_FILE, 'w') as f:
                    f.write(f"Validation Status: {validation_status}\n")
            else:
                validation_status = False
                with open(self.config.STATUS_FILE, 'w') as f:
                    f.write(f"Validation Status: {validation_status}\n")

            return validation_status
        except Exception as e:
            logger.exception(e)


