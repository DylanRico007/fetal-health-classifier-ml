import numpy as np

#Hacer una Matriz de Confusión
def matrizConfusion(y_real, y_pred, n_clases):
    matriz = np.zeros((n_clases, n_clases), dtype=int)
    for r, p in zip(y_real, y_pred):
        matriz[int(r)][int(p)] += 1
    return matriz

#Hacer Recall TP/(TP+FN)
def recall(matriz_conf):
    recalls = []
    for i in range(len(matriz_conf)):
        vp = matriz_conf[i, i]
        total_real_clase = np.sum(matriz_conf[i, :])
        if total_real_clase == 0:
            recalls.append(0.0)
        else:
            recalls.append(vp / total_real_clase)
    return np.array(recalls)


# 3. Función para calcular F1-Score por clase
def calcular_f1_score(matriz_conf):
    f1_scores = []
    for i in range(len(matriz_conf)):
        vp = matriz_conf[i, i]
        total_real_clase = np.sum(matriz_conf[i, :])  +
        total_pred_clase = np.sum(matriz_conf[:, i]) 

        recall = vp / total_real_clase if total_real_clase > 0 else 0.0
        precision = vp / total_pred_clase if total_pred_clase > 0 else 0.0

        if (precision + recall) == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * (precision * recall) / (precision + recall))
    return np.array(f1_scores)


