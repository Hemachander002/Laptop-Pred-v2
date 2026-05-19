from src.laptop_price_prediction.components.model_evaluation import ModelEvaluator
from src.laptop_price_prediction.config.configuration import ConfigurationManager
from src.laptop_price_prediction import logger


STAGE_NAME = "Model Evaluation Stage"

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def InitiateModelEvaluation(self):
        config = ConfigurationManager()
        model_eval_config = config.get_model_evaluation_config()
        model_eval = ModelEvaluator(config=model_eval_config)
        model_eval.log_into_mlflow()
    
if __name__ == "__main__":
    try:
        model_eval_pipeline = ModelEvaluationPipeline()
        model_eval_pipeline.InitiateModelEvaluation()
    except Exception as e:
        raise e