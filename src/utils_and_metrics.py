import numpy as np

# OPTIMIZACIÓN Y PREPROCESAMIENTO


def gradient_descent(grad_fn, x0, learning_rate=0.1, n_iterations=100):
    """Aplica descenso de gradiente estándar sobre una función objetivo."""
    x = np.array(x0, dtype=float)
    for _ in range(n_iterations):
        x = x - learning_rate * grad_fn(x)
    return x


def train_test_split(X, y, test_size=0.25, seed=0):
    """Divide matrices X e y en subconjuntos de entrenamiento y prueba."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    n_test = int(round(len(X) * test_size))
    test_idx, train_idx = indices[:n_test], indices[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def standardize(X):
    """Aplica escalado Z-Score estándar por columna."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std, mean, std


def k_fold_indices(n_samples, k=5, seed=0):
    """Genera particiones para validación cruzada k-fold."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    folds = np.array_split(indices, k)
    splits = []
    for i in range(k):
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        splits.append((train_idx, folds[i]))
    return splits

# MÉTRICAS DE CLASIFICACIÓN


def accuracy(y_true, y_pred):
    """Calcula la exactitud global de las predicciones."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred, n_classes=None):
    """
    Construye la matriz de confusión (filas: clase real, columnas: clase predicha).
    Requiere que las etiquetas estén indexadas en el rango [0, n_classes - 1].
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    
    if n_classes is None:
        n_classes = max(y_true.max(), y_pred.max()) + 1

    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for r, p in zip(y_true, y_pred):
        matrix[r, p] += 1
    return matrix


def precision(matrix):
    """Calcula precisión por clase a partir de la matriz de confusión: VP / (VP + FP)."""
    precisions = []
    for i in range(len(matrix)):
        vp = matrix[i, i]
        total_pred = np.sum(matrix[:, i])
        precisions.append(vp / total_pred if total_pred > 0 else 0.0)
    return np.array(precisions)


def recall(matrix):
    """Calcula sensibilidad (recall) por clase a partir de la matriz de confusión: VP / (VP + FN)."""
    recalls = []
    for i in range(len(matrix)):
        vp = matrix[i, i]
        total_real = np.sum(matrix[i, :])
        recalls.append(vp / total_real if total_real > 0 else 0.0)
    return np.array(recalls)


def f1_score(matrix):
    """Calcula F1-Score por clase a partir de la matriz de confusión."""
    precisions = precision(matrix)
    recalls = recall(matrix)
    f1_scores = []
    for p, r in zip(precisions, recalls):
        if (p + r) == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * (p * r) / (p + r))
    return np.array(f1_scores)