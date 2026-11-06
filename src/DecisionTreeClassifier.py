import numpy as np

class DecisionTreeClassifier:
    """
    --Árbol de Desición (Clasificación) para utilizar con el Bosque Aleatorio--
    
    Parámetros que recibe el árbol:
    
        - max_depth (int, por defecto=10):
            Profundidad máxima de niveles que puede alcanzar el árbol desde la raíz.
            Controla la complejidad del modelo y evita el sobreajuste.
        - min_samples_split (int, por defecto=2):
            Cantidad mínima de muestras que debe tener un nodo para permitir dividirlo.
            Si un nodo tiene menos datos que este, se convierte en nodo hoja.
        - max_features (int o None, por defecto=None):
            Cantidad de columnas o variables elegidas al azar que se evalúan en cada
            división. Si es None, evalúa todas las variables, si recibe un entero
            (como en el Random Forest), decorrelaciona los árboles del bosque.
        - rng (np.random.Generator o None, por defecto=None):
            Generador de números aleatorios de NumPy para seleccionar al azar las
            características cuando max_features está activo, asegurando reproducibilidad.

    Parámetros de datos que reciben los métodos de entrenamiento y predicción:

        - X (forma [n_samples, n_features]):
            Matriz con las variables predictoras numéricas de cada muestra.
        - y (forma [n_samples]):
            Vector con las etiquetas de clase codificadas como enteros desde 0 hasta K-1.
        - depth (int):
            Nivel de profundidad actual durante la construcción recursiva del árbol.
        - x (forma [n_features]):
            Vector que representa una única fila o muestra individual a clasificar.
        - node (dict):
            Diccionario anidado que representa el nodo actual del árbol (raíz, rama u hoja).
    """
    #Constructor e inicialización de hiperparámetros del árbol
    def __init__(self, max_depth=10, min_samples_split=2, max_features=None, rng=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.tree_ = None
    
    #Cálculo de la impureza de Gini de un nodo
    def _gini(self, y):
        proportions = np.bincount(y) / len(y)
        return 1.0 - np.sum(proportions ** 2)

    #Búsqueda voraz (greedy) de la mejor variable y umbral de corte
    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        features = np.arange(n_features)
        if self.max_features is not None:
            features = self.rng.choice(n_features, size=min(self.max_features, n_features), replace=False)
        best_feature, best_threshold, best_gini = None, None, np.inf
        for feature in features:
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                mask = X[:, feature] <= threshold
                if mask.all() or not mask.any():
                    continue
                gini = (mask.sum() * self._gini(y[mask]) + (~mask).sum() * self._gini(y[~mask])) / n_samples
                if gini < best_gini:
                    best_feature, best_threshold, best_gini = feature, threshold, gini
        return best_feature, best_threshold

    #Construcción recursiva del árbol de decisión
    def _build(self, X, y, depth):
        if len(np.unique(y)) == 1 or depth >= self.max_depth or len(y) < self.min_samples_split:
            return {'leaf': True, 'class': np.bincount(y).argmax()}
        feature, threshold = self._best_split(X, y)
        if feature is None:
            return {'leaf': True, 'class': np.bincount(y).argmax()}
        mask = X[:, feature] <= threshold
        return {
            'leaf': False,
            'feature': feature,
            'threshold': threshold,
            'left': self._build(X[mask], y[mask], depth + 1),
            'right': self._build(X[~mask], y[~mask], depth + 1)
        }

    #Método de entrenamiento del árbol
    def fit(self, X, y):
        self.tree_ = self._build(X, y, 0)
        return self

    #Predicción iterativa para una sola muestra individual
    def _predict_one(self, x, node):
        while not node['leaf']:
            node = node['left'] if x[node['feature']] <= node['threshold'] else node['right']
        return node['class']

    #Método de predicción para un conjunto de muestras
    def predict(self, X):
        return np.array([self._predict_one(x, self.tree_) for x in X])
