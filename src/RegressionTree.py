import numpy as np

class RegressionTree:
    """
    --Árbol de regresión base para Gradient Boosting (predicen residuos continuos, errores de probabilidad)--

    Parámetros que recibe el árbol al inicializarse:
    
        - max_depth (int, por defecto=3):
            Profundidad máxima de niveles que puede alcanzar el árbol desde la raíz.
            Limita las interacciones entre variables y mantiene al árbol como un aprendiz débil.
        - min_samples_split (int, por defecto=2):
            Cantidad mínima de muestras que debe contener un nodo para permitir dividirlo.
            Si tiene menos muestras que este umbral, el nodo se vuelve una hoja.
        - max_thresholds (int o None, por defecto=32):
            Número máximo de puntos de corte (cuantiles) a evaluar por cada característica
            cuando una columna continua tiene muchos valores únicos. Acelera el entrenamiento.

    Parámetros que reciben los métodos de construcción y predicción:

        - X (forma [n_samples, n_features]):
            Matriz con las características predictoras numéricas de las muestras.
        - y (forma [n_samples]):
            Vector con los valores continuos objetivo (los residuos de probabilidad en Gradient Boosting).
        - depth (int):
            Nivel de profundidad actual durante la llamada recursiva de construcción del árbol.
        - x (forma [n_features]):
            Vector correspondiente a una sola fila o muestra individual que se desea predecir.
        - node (dict):
            Diccionario que representa el nodo actual del árbol (nodo interno de corte o nodo hoja).
    """
    
    #Constructor e inicialización de hiperparámetros del árbol de regresión
    def __init__(self, max_depth=3, min_samples_split=2, max_thresholds=32):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_thresholds = max_thresholds  
        self.tree_ = None

    #Búsqueda voraz de la característica y umbral que minimizan el error cuadrático (SSE)
    def _best_split(self, X, y):
        best_feature, best_threshold, best_sse = None, None, np.inf
        for feature in range(X.shape[1]):
            uniq = np.unique(X[:, feature])[:-1]
            if len(uniq) == 0:
                continue
            if self.max_thresholds is not None and len(uniq) > self.max_thresholds:
                idx = np.linspace(0, len(uniq) - 1, self.max_thresholds, dtype=int)
                thresholds = uniq[idx]
            else:
                thresholds = uniq

            for threshold in thresholds:
                mask = X[:, feature] <= threshold
                left, right = y[mask], y[~mask]
                sse = np.sum((left - left.mean()) ** 2) + np.sum((right - right.mean()) ** 2)
                if sse < best_sse:
                    best_feature, best_threshold, best_sse = feature, threshold, sse
        return best_feature, best_threshold

    #Construcción recursiva de la estructura del árbol de regresión
    def _build(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples_split or np.all(y == y[0]):
            return {'leaf': True, 'value': y.mean()}
        feature, threshold = self._best_split(X, y)
        if feature is None:
            return {'leaf': True, 'value': y.mean()}
        mask = X[:, feature] <= threshold
        return {
            'leaf': False,
            'feature': feature,
            'threshold': threshold,
            'left': self._build(X[mask], y[mask], depth + 1),
            'right': self._build(X[~mask], y[~mask], depth + 1),
        }

    #Método de entrenamiento del árbol (los construye de forma recursiva)
    def fit(self, X, y):
        self.tree_ = self._build(X, y, 0)
        return self

    #Recorrido iterativo del árbol para predecir una única muestra
    def _predict_one(self, x, node):
        while not node['leaf']:
            node = node['left'] if x[node['feature']] <= node['threshold'] else node['right']
        return node['value']

    #Método público de predicción para múltiples muestras 
    def predict(self, X):
        return np.array([self._predict_one(x, self.tree_) for x in X])