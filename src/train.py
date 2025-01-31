from src.data_processing import DataProcessor
from src.model import XGBoostModel
from src.tracking import MLflowTracker




# Caminho dos dados
data_path = "data/water_potability.csv"

# Processamento de dados
processor = DataProcessor(data_path)
processor.load_data()  # Certifica que os dados foram carregados
X_train, X_test, y_train, y_test = processor.preprocess()

# Treinar o modelo
model = XGBoostModel()
model.train(X_train, y_train)

# Avaliar o modelo
accuracy = model.evaluate(X_test, y_test)

# Criar tracking no MLflow
tracker = MLflowTracker()
tracker.log_model(model.model, accuracy)

# Salvar modelo localmente
model.save_model("models/xgboost_water.json")
