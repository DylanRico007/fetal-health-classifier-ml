import numpy as np
from DecisionTreeClassifier import DecisionTreeClassifier

class RandomForestClassifier:
    """
    --Bosque Aleatorio de clasificación multiclase--
    
    Parámetros del constructor del bosque (__init__):

        - n_trees (int, por defecto=50):
            Número total de árboles de decisión independientes que conformarán el bosque.
        - max_depth (int, por defecto=10):
            Profundidad máxima que se le pasará como parámetro a cada DecisionTreeClassifier.
        - min_samples_split (int, por defecto=2):
            Mínimo de muestras para dividir un nodo que se le pasará a cada DecisionTreeClassifier.
        - seed (int, por defecto=0):
            Semilla maestra para inicializar el generador aleatorio del muestreo Bootstrap y de los árboles.
    """

    #Constructor e inicialización del estado del bosque
    def __init__(self, n_trees=50, max_depth=10, min_samples_split=2, seed=0):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.seed = seed
        self.trees_ = []
        self.oob_indices_ = []
        self.classes_ = None
        self.n_classes_ = None

    #Generación de muestras Bootstrap (muestreo aleatorio con reemplazo)
    def _bootstrap_sample(self, X, y, rng):
        n = len(X)
        indices = rng.integers(0, n, size=n)
        return X[indices], y[indices], indices

    #Entrenamiento del Bosque Aleatorio
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)

        rng = np.random.default_rng(self.seed)
        max_features = max(1, int(np.sqrt(X.shape[1])))
        self.trees_ = []
        self.oob_indices_ = []

        for _ in range(self.n_trees):
            X_b, y_b, idx = self._bootstrap_sample(X, y_encoded, rng)
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_features,
                rng=rng,
            )
            tree.fit(X_b, y_b)
            self.trees_.append(tree)
            self.oob_indices_.append(np.setdiff1d(np.arange(len(X)), idx))
        return self

    #Predicción del bosque por votación mayoritaria 
    def predict(self, X):
        X = np.asarray(X, dtype=float)
        all_votes = np.array([tree.predict(X) for tree in self.trees_])
        majority_idx = np.array([
            np.bincount(col, minlength=self.n_classes_).argmax()
            for col in all_votes.T
        ])
        return self.classes_[majority_idx]

    #Evaluación interna Out-Of-Bag (validación sin el conjunto de prueba externo)
    def oob_score(self, X, y):
        X = np.asarray(X, dtype=float)
        _, y_encoded = np.unique(y, return_inverse=True)
        votes = np.zeros((len(X), self.n_classes_), dtype=int)
        for tree, oob in zip(self.trees_, self.oob_indices_):
            if len(oob) == 0:
                continue
            preds = tree.predict(X[oob])
            votes[oob, preds] += 1
        covered = votes.sum(axis=1) > 0
        return np.mean(votes[covered].argmax(axis=1) == y_encoded[covered])
