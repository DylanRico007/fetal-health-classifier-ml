import numpy as np
from collections import Counter


class KNNClassifier:
    """
    --Clasificador K-Nearest Neighbors (KNN) basado en distancia euclidiana--

    Parámetros que recibe el clasificador al inicializarse:

        - n_neighbors (int, por defecto=3):
            Número de vecinos más cercanos que se consultan para determinar la clase
            de una nueva muestra mediante votación mayoritaria.

    Parámetros de datos que reciben los métodos internos y de entrenamiento/predicción:

        - X (forma [n_samples, n_features]):
            Matriz con las variables predictoras numéricas estandarizadas de cada muestra.
        - y (forma [n_samples]):
            Vector con las etiquetas de clase correspondientes a cada muestra.
        - a, b (forma [n_features]):
            Vectores individuales entre los cuales se calcula la distancia euclidiana.
        - x (forma [n_features]):
            Vector que representa una única muestra individual a clasificar.
    """

    # Constructor e inicialización de hiperparámetros del modelo
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    # Almacenamiento del conjunto de entrenamiento (aprendizaje perezoso / lazy learning)
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self

    # Cálculo de la distancia euclidiana entre dos vectores de características
    def _euclidean_distance(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    # Predicción por votación mayoritaria de los K vecinos más cercanos para una muestra
    def _predict_one(self, x):
        distances = np.array([self._euclidean_distance(x, x_train) for x_train in self.X_train])
        nearest_indices = np.argsort(distances)[:self.n_neighbors]
        nearest_labels = self.y_train[nearest_indices]
        return Counter(nearest_labels.tolist()).most_common(1)[0][0]
    
    # Método de predicción de probabilidades para ROC/AUC
    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_proba_one(x) for x in X])

    # Método de predicción para un conjunto de muestras
    def predict(self, X):
        return np.array([self._predict_one(x) for x in X])
