import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class XGBoostModel:
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )

    def train(self, X_train, y_train):
        """Treina o modelo."""
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """Avalia o modelo com múltiplas métricas."""
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"Acurácia: {acc:.4f}")
        print(f"Precisão: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")

        return acc, precision, recall, f1

    def save_model(self, model_path: str):
        """Salva o modelo treinado."""
        self.model.save_model(model_path)
