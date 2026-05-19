import pandas as pd
from xgboost import XGBRegressor
from src.laptop_price_prediction.config.configuration import ConfigurationManager
from src.laptop_price_prediction.entity.config_entity import (ModelTrainerConfig)
from src.laptop_price_prediction import logger
import joblib
import os
import dagshub

os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/Hemachander002/Laptop-Pred-v2.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "Hemachander002"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "254edea09d48445e8e8de0075ef293dfc5f9e30a"

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train_model(self):
        logger.info("Loading training data...")
        train_data = pd.read_csv(self.config.train_data_path)
        X_train = train_data.drop(columns=[self.config.target_column],axis = 1)
        y_train = train_data[[self.config.target_column]]

        logger.info("Loading testing data...")
        test_data = pd.read_csv(self.config.test_data_path)
        X_test = test_data.drop(columns=[self.config.target_column],axis = 1)
        y_test = test_data[[self.config.target_column]]

        logger.info("Training the XGBoost model...")
        model = XGBRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            min_child_weight=self.config.min_child_weight,
            gamma=self.config.gamma,
            random_state=self.config.random_state
        )
        model.fit(X_train, y_train)

        logger.info("Saving the trained model...")
        joblib.dump(model, os.path.join(self.config.root_dir, self.config.model_name))
