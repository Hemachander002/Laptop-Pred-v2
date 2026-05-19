import joblib
import dagshub
import mlflow
from mlflow.metrics import mae
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from urllib.parse import urlparse
from src.laptop_price_prediction.entity.config_entity import ModelEvaluationConfig
from src.laptop_price_prediction.utils.common import save_json
from src.laptop_price_prediction.config.configuration import ConfigurationManager, ModelEvaluationConfig
from pathlib import Path
import numpy as np
import pandas as pd


class ModelEvaluator:
    def __init__(self,config: ModelEvaluationConfig):
        self.config = config
        self.model_path = config.model_path
        self.X_test = config.test_data_path
        self.y_test = config.target_column
        self.registry_uri = config.mlflow_uri
        self.metric_file_name = config.metric_file_name

    def eval_metrics(self,actual, predicted):
        mse = mean_squared_error(actual, predicted)
        r2 = r2_score(actual, predicted)
        rootmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, predicted)

        return mse, r2, rootmse, mae

    def log_into_mlflow(self):
        test_data = pd.read_csv(self.X_test)
        y_test = test_data[self.y_test]
        X_test = test_data.drop(columns=[self.y_test])
        mlflow.set_registry_uri(self.registry_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        model = joblib.load(self.model_path)

        with mlflow.start_run():
            predicted = model.predict(X_test)
            mse, r2, rootmse, mae = self.eval_metrics(y_test, predicted)
            scores = {"mse": mse, "r2": r2, "rootmse": rootmse, "mae": mae}
            save_json(path= Path(self.metric_file_name), data=scores)
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(scores)

            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(model, name =  "model", registered_model_name="XGBoostRegressor")
            else:
                mlflow.sklearn.log_model(model, name =  "model")


if __name__ == "__main__":
    try:
        config = ConfigurationManager()
        model_eval_config = config.get_model_evaluation_config()
        model_eval = ModelEvaluator(config=model_eval_config)
        model_eval.log_into_mlflow()
    except Exception as e:
        raise e
    