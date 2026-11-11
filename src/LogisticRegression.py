import numpy as np

class LogisticRegression:

    """""
    Regresion logistica / logistic regression

    Hiper_Parametros del modelo:
    - Learning rate = 0.1
        tasa de aprendizaje que controla los tamaños del paso del descenso de gradiente
    - n_iterations = 1000
        iteraciones a realizar para la optimizacion de base
    - threshold (float, por defecto=0.3):
        Umbral de probabilidad para asignar la clase 1.
        Por defecto = 0.3, pero en este caso un ajuste de 0.3 puede ser más adecuado para predecir mejor la clase 1
    
    Parámetros de datos que reciben los métodos de entrenamiento y predicción:

    - X (forma [n_samples, n_features]):
        Matriz de características predictoras numéricas.
    - y (forma [n_samples]):
        Vector con las etiquetas de clase codificadas como 0 o 1.
    - weights (forma [n_features]):
        Vector de pesos del modelo, uno por cada característica.
    - biases (float):
        Sesgo o intercepto del modelo.
    
    """""

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
import numpy as np

class LogisticRegression:

    """""
    Regresion logistica / logistic regression

    Hiper_Parametros del modelo:
    - Learning rate = 0.1
        tasa de aprendizaje que controla los tamaños del paso del descenso de gradiente
    - n_iterations = 1000
        iteraciones a realizar para la optimizacion de base
    - threshold (float, por defecto=0.3):
        Umbral de probabilidad para asignar la clase 1.
        Por defecto = 0.3, pero en este caso un ajuste de 0.3 puede ser más adecuado para predecir mejor la clase 1
    
    Parámetros de datos que reciben los métodos de entrenamiento y predicción:

    - X (forma [n_samples, n_features]):
        Matriz de características predictoras numéricas.
    - y (forma [n_samples]):
        Vector con las etiquetas de clase codificadas como 0 o 1.
    - weights (forma [n_features]):
        Vector de pesos del modelo, uno por cada característica.
    - biases (float):
        Sesgo o intercepto del modelo.
    
    """""

    #Inicialización de hiperparámetros y pesos del modelo
    def __init__(self, learning_rate = 0.1, n_iterations = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.biases = None

    #funcion activacion para transformar un valor numero en probabilidad de 0 y 1
    def sigmoide(self, z):
        z = np.clip(z, -500, 500) #clipping para evitar overflow en la exponencial
        return 1 / (1 + np.exp(-z))

    #Predice la probabilidad de que cada muestra pertenezca a la clase 1
    def predict_proba(self, X):
        return self.sigmoide(X.dot(self.weights) + self.biases)

    #Clasifica cada muestra en clase 0 o 1 según el umbral de probabilidad
    def predict(self, X, threshold=0.25):
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