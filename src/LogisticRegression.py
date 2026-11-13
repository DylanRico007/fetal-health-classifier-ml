import numpy as np

class LogisticRegression:

    """
    --Regresión Logística Binaria optimizada mediante Descenso de Gradiente--

    Parámetros que recibe el clasificador al inicializarse:

        - learning_rate (float, por defecto=0.1):
            Tasa de aprendizaje que controla el tamaño del paso en cada actualización del gradiente.
        - n_iterations (int, por defecto=1000):
            Número de iteraciones completas sobre el conjunto de entrenamiento para optimizar pesos y sesgo.

    Parámetros de datos que reciben los métodos de entrenamiento y predicción:

        - X, x (forma [n_samples, n_features]):
            Matriz de características predictoras numéricas estandarizadas.
        - y (forma [n_samples]):
            Vector binario con las etiquetas de clase codificadas como 0 o 1.
        - z (forma [n_samples]):
            Combinación lineal (X · weights + biases) evaluada por la función sigmoide.
        - threshold (float, por defecto=0.3):
            Umbral de probabilidad utilizado en predict() para asignar la clase positiva (1).
    """

    #Inicialización de hiperparámetros y pesos del modelo
    def __init__(self, learning_rate = 0.1, n_iterations = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.biases = None

    #funcion activacion para transformar un valor numero en probabilidad de 0 y 1
    def sigmoide(self, z):
        return 1 / (1 + np.exp(-z))

    #Predice la probabilidad de que cada muestra pertenezca a la clase 1
    def predict_proba(self, X):
        return self.sigmoide(X.dot(self.weights) + self.biases)

    #Clasifica cada muestra en clase 0 o 1 según el umbral de probabilidad
    def predict(self, X, threshold=0.3):
        return (self.predict_proba(X) >= threshold).astype(int)

    #Calcula la función de costo logístico (log loss) para evaluar el rendimiento del modelo
    def compute_cost(self, X, y):
        eps = 1e-15 #valor pequeño para evitar log(0)
        p = np.clip(self.predict_proba(X), eps, 1 - eps)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    #Calcula los gradientes para la optimización
    def compute_gradients(self, X, y):
        n_samples = X.shape[0] 
        errors = self.predict_proba(X) - y
        dw = (1 / n_samples) * X.T.dot(errors)
        db = (1 / n_samples) * np.sum(errors)
        return dw, db

    #Entrenamiento por gradiente
    def fit(self, x, y):
        n_features = x.shape[1]
        self.weights = np.zeros(n_features)
        self.biases = 0.0
        for _ in range(self.n_iterations):
            dw, db = self.compute_gradients(x, y)
            self.weights -= self.learning_rate * dw
            self.biases -= self.learning_rate * db
        return self