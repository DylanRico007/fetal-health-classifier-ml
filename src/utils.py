import numpy as np

def gradient_descent(grad_fn, x0, learning_rate=0.1, n_iterations=100):
    x = np.array(x0, dtype=float)
    for _ in range(n_iterations):
        x = x - learning_rate * grad_fn(x)
    return x

def train_test_split(X, y, test_size=0.25, seed=0):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    n_test = int(round(len(X) * test_size))
    test_idx, train_idx = indices[:n_test], indices[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def standardize(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std, mean, std

def k_fold_indices(n_samples, k=5, seed=0):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    folds = np.array_split(indices, k)
    splits = []
    for i in range(k):
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        splits.append((train_idx, folds[i]))
    return splits

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def confusion_counts(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    return tp, fp, fn, tn

def precision_recall_f1(y_true, y_pred):
    tp, fp, fn, tn = confusion_counts(y_true, y_pred)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1