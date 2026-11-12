import numpy as np
class KNNClassifier: 
  """
    --K-Nearest Neighbors (KNN) para Clasificación--

    Algoritmo de aprendizaje supervisado que clasifica nuevas muestras
    basándose en las clases de sus K vecinos más cercanos dentro del
    conjunto de entrenamiento, utilizando la distancia euclidiana.

    Parámetros que recibe el clasificador:

        - k (int, por defecto=5):
            Cantidad de vecinos más cercanos que se consideran para
            determinar la clase de una nueva muestra.
            La clasificación se realiza mediante una votación
            mayoritaria entre las etiquetas de estos vecinos.

    Parámetros de datos que reciben los métodos de entrenamiento y predicción:

        - X (forma [n_samples, n_features]):
            Matriz con las variables predictoras numéricas de cada muestra.
            Durante el entrenamiento almacena los datos conocidos y
            durante la predicción contiene las muestras a clasificar.

        - y (forma [n_samples]):
            Vector con las etiquetas de clase codificadas como enteros
            no negativos, donde cada valor representa una categoría.

    Atributos internos del clasificador:

        - X_train:
            Matriz que almacena las características de las muestras
            utilizadas durante el entrenamiento.

        - y_train:
            Vector que almacena las etiquetas correspondientes a
            las muestras de entrenamiento.

        - distancias:
            Vector con las distancias euclidianas entre una muestra
            nueva y todas las muestras del conjunto de entrenamiento.

        - k_indices:
            Índices de las K muestras con menor distancia respecto
            a la muestra que se desea clasificar.

        - k_labels:
            Etiquetas de clase correspondientes a los K vecinos
            más cercanos seleccionados.

        - votos:
            Conteo de las veces que aparece cada clase entre
            los vecinos seleccionados.

        - preds:
            Lista que almacena las predicciones realizadas antes
            de convertirlas en un arreglo de NumPy.
    """ 
def __init__(self, k=5):
    self.k = k
        
def fit(self, X, y):
    self.X_train = np.array(X)
    self.y_train = np.array(y)
    
def predict(self, X):
    X = np.array(X)
    preds = []
    for x in X:
        distancias = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        k_indices = np.argsort(distancias)[:self.k]
        k_labels = self.y_train[k_indices].astype(int)
        votos = np.bincount(k_labels)
        preds.append(np.argmax(votos))
    return np.array(preds)

