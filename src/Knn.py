import numpy as np    

class KNNClassifier:
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
