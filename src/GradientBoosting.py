import numpy as np
from .RegressionTree import RegressionTree

class GradientBoostingClassifier:
    """
    --Adaptación de Gradient Boosting para Clasificación Multiclase--

    Parámetros que recibe el clasificador al inicializarse:

        - n_estimators (int, por defecto=50):
            Número de etapas secuenciales de boosting (en cada etapa se entrena 1 árbol por clase).
        - learning_rate (float, por defecto=0.1):
            Tasa de aprendizaje que multiplica la aportación de cada árbol para evitar sobreajuste.
        - max_depth (int, por defecto=3):
            Profundidad máxima que se le pasará como parámetro a cada RegressionTree individual.
        - balanced (bool, por defecto=True):
            Activa la ponderación inversa a la frecuencia de cada clase (sample_weights) e inicia
            con probabilidades neutras para compensar el desbalance de datos.

    Parámetros que reciben los métodos internos y de entrenamiento/predicción:

        - F (forma [n_samples, n_classes]):
            Matriz con las predicciones acumuladas en escala log-odds.
        - Y_onehot (forma [n_samples, n_classes]):
            Matriz binaria codificada en One-Hot donde cada fila tiene un 1.0 en la columna de su clase real y 0.0 en las demás.
        - X (forma [n_samples, n_features]):
            Matriz de variables predictoras del dataset.
        - y (forma [n_samples]):
            Vector con las etiquetas originales de clase.
    """

    #Constructor e inicialización de hiperparámetros del Gradient Boosting
    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=3, balanced=True):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.balanced = balanced
        self.init_ = None
        self.trees_ = []
        self.classes_ = None
        self.n_classes_ = None


    #Función de activación Softmax para transformar puntajes continuos F en probabilidades
    def _softmax(self, F):
        exp_F = np.exp(F - np.max(F, axis=1, keepdims=True))
        return exp_F / np.sum(exp_F, axis=1, keepdims=True)


    #Cálculo de la predicción base inicial antes de entrenar los árboles
    def _initial_prediction(self, Y_onehot):
        #Logaritmo de la proporción de cada clase en el dataset
        p = np.clip(Y_onehot.mean(axis=0), 1e-15, 1.0)
        return np.log(p)


    #Cálculo de los pseudo-residuos 
    def _residuals(self, Y_onehot, F):
        #Residuo de probabilidad. La etiqueta real (0 o 1) - probabilidad predicha
        return Y_onehot - self._softmax(F)


    #Entrenamiento secuencial del modelo multiclase
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        #Codifica automáticamente las clases [1.0, 2.0, 3.0] a índices [0, 1, 2]
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        Y_onehot = np.eye(self.n_classes_)[y_idx]   # Crea la matriz One-Hot de dimensión (n_samples, n_classes) 

        if self.balanced:  # Verifica si se activó el balanceo de clases
            self.init_ = np.zeros(self.n_classes_)  # Arranca con probabilidades iguales (neutras) para todas las clases
            conteo = np.bincount(y_idx)  # Cuenta cuántos pacientes hay en cada clase
            pesos_clase = len(y_idx) / (self.n_classes_ * conteo)  # Calcula peso inversamente proporcional a su frecuencia
            sample_weights = pesos_clase[y_idx][:, None]  # Asigna a cada paciente el peso de su clase real
        else:  # Mantiene el flujo original si balanced=False
            self.init_ = self._initial_prediction(Y_onehot)
            sample_weights = 1.0  # Asigna peso neutro (1.0) a todos por igual
            
        F = np.tile(self.init_, (len(X), 1))
        self.trees_ = []

        for _ in range(self.n_estimators):
            residuals = self._residuals(Y_onehot, F) * sample_weights  # Pondera los errores de probabilidad según el peso de la clase
            stage_trees = []
            # Entrena un RegressionTree por cada una de las K clases
            for k in range(self.n_classes_):
                tree = RegressionTree(max_depth=self.max_depth).fit(X, residuals[:, k])
                F[:, k] += self.learning_rate * tree.predict(X)
                stage_trees.append(tree)
            self.trees_.append(stage_trees)
        return self


    #Predicción de probabilidades de pertenencia a cada clase
    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        F = np.tile(self.init_, (X.shape[0], 1))
        for stage_trees in self.trees_:
            for k, tree in enumerate(stage_trees):
                F[:, k] += self.learning_rate * tree.predict(X)
        return self._softmax(F)


    #Predicción de la etiqueta de clase final para nuevas muestras
    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


    #Predicción paso a paso por etapa para analizar la evolución del modelo
    def staged_predict(self, X):
        X = np.asarray(X, dtype=float)
        F = np.tile(self.init_, (X.shape[0], 1))
        stages = []
        for stage_trees in self.trees_:
            for k, tree in enumerate(stage_trees):
                F[:, k] += self.learning_rate * tree.predict(X)
            preds_stage = self.classes_[np.argmax(self._softmax(F), axis=1)]
            stages.append(preds_stage)
        return stages
