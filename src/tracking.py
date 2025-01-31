import mlflow
import mlflow.xgboost
import xgboost as xgb

class MLflowTracker:
    def __init__(self, experiment_name="WaterPotability"):
        """Configura o MLflow para registrar experimentos e logs."""
        mlflow.set_tracking_uri("file:///C:/Users/FKA9/OneDrive - PETROBRAS/Área de Trabalho/MLP/ml-class-agua/mlruns")  # Definir caminho do MLflow manualmente
        mlflow.set_experiment(experiment_name)

    def log_model(self, model: xgb.XGBClassifier, accuracy: float):
        """Registra o modelo e métricas no MLflow."""
        with mlflow.start_run():
            mlflow.log_metric("accuracy", accuracy)
            mlflow.xgboost.log_model(model, artifact_path="models")
            print("Modelo registrado no MLflow com sucesso!")
