import numpy as np
import matplotlib.pyplot as plt

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
    idx_train_end = int((1 - test_size) * len(X))
    train_idx = indices[:idx_train_end]
    test_idx = indices[idx_train_end:]
    
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

def calcular_roc_auc(y_true_bin, y_scores):
    """
    Calcula FPR, TPR y AUC para una clase binarizada (Uno contra el resto)
    """
    #Ordenar muestras de mayor a menor probabilidad predicha
    desc_idx = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true_bin[desc_idx]
    y_scores_sorted = y_scores[desc_idx]

    # Detectar los índices donde cambia el valor de probabilidad 
    distinct_value_indices = np.where(np.diff(y_scores_sorted))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true_sorted.size - 1]

    # Acumular Verdaderos Positivos (TP) y Falsos Positivos (FP) por umbral
    tps = np.cumsum(y_true_sorted)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    # Agregar el origen (0, 0) de la gráfica
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    # Normalizar a tasas y calcular área bajo la curva (con la regla del regla del trapecio)
    tpr = tps / tps[-1]
    fpr = fps / fps[-1]
    auc = np.trapezoid(tpr, fpr) if hasattr(np, "trapezoid") else np.trapz(tpr, fpr)

    return fpr, tpr, float(auc)


def roc(model, X_test, y_test):
    """
    Genera la gráfica ROC One-vs-Rest para las 3 clases y AUC.
    """
    y_probas = model.predict_proba(X_test) #Se necesita de la predicción con probabilidades (este método es el que se encuentra en Random Forest)
    nombres = {1: "Normal (1)", 2: "Suspect (2)", 3: "Pathologic (3)"}
    colores = {1: "#2b5c8f", 2: "#e69f00", 3: "#d55e00"}

    plt.figure(figsize=(8, 6))
    aucs = []

    for c in [1, 2, 3]:
        # 1 si es la clase actual, 0 en otro caso
        y_bin = (y_test == c).astype(int)
        fpr, tpr, auc_val = calcular_roc_auc(y_bin, y_probas[:, c - 1])
        aucs.append(auc_val)

        plt.plot(
            fpr,
            tpr,
            color=colores[c],
            lw=2.2,
            label=f"{nombres[c]} (AUC = {auc_val:.4f})",
        )

    # Línea diagonal de referencia 
    plt.plot([0, 1], [0, 1], "k--", lw=1.2, label="Clasificador Aleatorio (AUC = 0.5)")

    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.03])
    plt.xlabel("Tasa de Falsos Positivos (FPR)", fontsize=11)
    plt.ylabel("Tasa de Verdaderos Positivos (TPR / Recall)", fontsize=11)
    plt.title(
        f"Curva ROC Multiclase (One-vs-Rest) - Random Forest\n AUC = {np.mean(aucs):.4f}",
        fontsize=12,
    )
    plt.legend(loc="lower right", frameon=True)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return np.mean(aucs), aucs