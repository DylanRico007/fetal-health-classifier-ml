"""Clasificador KNN propio para el proyecto de salud fetal.

El algoritmo utiliza distancia euclidiana y votación uniforme; únicamente
requiere NumPy. Las fracciones de votos de los vecinos se exponen mediante
``predict_proba`` para evaluar curvas ROC multiclase.
"""

import numpy as np


class KNNClassifier:
    """K vecinos más cercanos para clasificación multiclase.

    Parámetros
    ----------
    k : int, predeterminado 5
        Número de vecinos utilizados para clasificar.

    Atributos después de fit
    ------------------------
    X_train : ndarray, forma (n_muestras, n_características)
        Características del conjunto de entrenamiento.
    y_train : ndarray, forma (n_muestras,)
        Etiquetas originales del conjunto de entrenamiento.
    classes_ : ndarray, forma (n_clases,)
        Clases ordenadas; define el orden de columnas de predict_proba.
    """

    def __init__(self, k=5):
        if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError("k debe ser un entero positivo.")
        self.k = int(k)

    def fit(self, X, y):
        """Almacena los datos etiquetados para búsquedas posteriores."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("X debe ser bidimensional e y un vector unidimensional.")
        if len(X) == 0 or len(X) != len(y):
            raise ValueError("X e y deben tener la misma cantidad positiva de filas.")
        if not np.all(np.isfinite(X)):
            raise ValueError("X debe contener únicamente valores numéricos finitos.")
        if self.k > len(X):
            raise ValueError("k no puede superar el número de muestras de entrenamiento.")

        self.X_train = X.copy()
        self.y_train = y.copy()
        self.classes_, self._y_indices = np.unique(self.y_train, return_inverse=True)
        return self

    def predict_proba(self, X):
        """Fracción de los k vecinos que pertenece a cada clase.

        Devuelve un arreglo (n_muestras, n_clases). Cada fila suma 1, y
        las columnas siguen el orden de ``self.classes_``. Estas son
        puntuaciones empíricas por votos, no probabilidades calibradas.
        """
        if not hasattr(self, "X_train"):
            raise ValueError("Primero se debe llamar al método fit.")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.X_train.shape[1]:
            raise ValueError("X debe ser una matriz con el mismo número de características que X_train.")
        if not np.all(np.isfinite(X)):
            raise ValueError("X debe contener únicamente valores numéricos finitos.")

        probabilidades = np.zeros((len(X), len(self.classes_)), dtype=float)
        for i, x in enumerate(X):
            # La raíz cuadrada no cambia el orden de las distancias euclidianas.
            distancias_cuadradas = np.sum((self.X_train - x) ** 2, axis=1)
            indices_vecinos = np.argsort(distancias_cuadradas)[:self.k]
            votos = np.bincount(self._y_indices[indices_vecinos], minlength=len(self.classes_))
            probabilidades[i] = votos / self.k
        return probabilidades

    def predict(self, X):
        """Predice la clase mayoritaria. En empates gana la menor etiqueta."""
        probabilidades = self.predict_proba(X)
        return self.classes_[np.argmax(probabilidades, axis=1)]
