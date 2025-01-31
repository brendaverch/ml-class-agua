import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

class DataProcessor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = None

    def load_data(self) -> pd.DataFrame:
        """Carrega os dados do arquivo CSV e verifica se está vazio."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {self.file_path}")

        self.data = pd.read_csv(self.file_path)

        if self.data is None or self.data.empty:
            raise ValueError("O arquivo de dados está vazio ou não foi carregado corretamente.")

        return self.data

    def preprocess(self):
        """Realiza o pré-processamento dos dados."""
        if self.data is None:
            raise ValueError("Os dados ainda não foram carregados. Chame load_data() antes de preprocess().")

        self.data = self.data.dropna()  # Remover valores nulos

        X = self.data.drop(columns=["Potability"])
        y = self.data["Potability"]

        # Normalização dos dados
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Divisão em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        return X_train, X_test, y_train, y_test  # 🔹 Corrigido: garantindo o retorno correto
