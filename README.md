# Clasificación del Estado Fetal mediante Cardiotocografía y Machine Learning

## Introducción: ¿por qué importa este problema?

Durante el embarazo y el trabajo de parto, el bienestar fetal es una preocupación fundamental para el personal de salud y las familias. Una de las herramientas utilizadas para vigilarlo es la **cardiotocografía (CTG)**, un procedimiento que registra la frecuencia cardiaca fetal y la actividad uterina.

Estos registros contienen patrones que pueden aportar información sobre el estado del feto. Sin embargo, su interpretación requiere experiencia clínica y debe considerar diferentes indicadores, como aceleraciones, desaceleraciones y variabilidad de la frecuencia cardiaca.

El problema que aborda este proyecto es la identificación de patrones asociados con tres estados fetales: normal, sospechoso y patológico. Esto resulta relevante para el seguimiento obstétrico, ya que una clasificación incorrecta puede dificultar la identificación de registros que requieren mayor atención.

El proyecto explora el uso de herramientas computacionales para analizar estas características y evaluar hasta qué punto pueden reproducir las clasificaciones de referencia realizadas por especialistas.

**¿A quién afecta este problema?**

- **Personas embarazadas:** son quienes reciben el seguimiento obstétrico y cuyos registros pueden requerir interpretación durante el embarazo o el parto.
- **Fetos:** su bienestar es el objeto principal de la monitorización.
- **Personal médico y obstétrico:** debe interpretar los registros y decidir si es necesaria una evaluación o intervención adicional.

Este trabajo no busca sustituir la valoración médica, sino estudiar una posible herramienta de apoyo experimental para la clasificación de registros cardiotocográficos.

## Antecedentes

La cardiotocografía se ha utilizado como método de vigilancia fetal mediante el registro de la frecuencia cardiaca y las contracciones uterinas. Con el desarrollo de sistemas computacionales, también se han generado herramientas para extraer características de estos registros y facilitar su análisis.

Uno de los conjuntos de datos utilizados en esta línea de investigación es **Cardiotocography Data Set**, disponible en UCI Machine Learning Repository. Sus registros fueron procesados automáticamente y clasificados por tres especialistas en obstetricia, quienes asignaron una etiqueta de consenso para cada caso.

Las categorías utilizadas son:

- **Normal:** registro clasificado dentro de los parámetros normales de referencia.
- **Sospechoso:** registro que no cumple todos los criterios para considerarse normal ni reúne los criterios de la categoría patológica.
- **Patológico:** registro que presenta características asociadas con alteraciones relevantes en la clasificación de referencia.

Este conjunto permite estudiar cómo diferentes métodos computacionales identifican patrones relacionados con el estado fetal.

## Objetivo del proyecto

Desarrollar y comparar modelos de clasificación capaces de identificar el estado fetal a partir de características previamente extraídas de registros cardiotocográficos.

Además de comparar su rendimiento general, se busca analizar su capacidad para reconocer las clases menos representadas, especialmente la categoría patológica.

La finalidad es evaluar las posibilidades y limitaciones de este enfoque, no establecer un sistema de diagnóstico clínico.

## Dataset

Se utiliza el conjunto de datos **Cardiotocography Data Set**, distribuido a través de UCI Machine Learning Repository.

- **Archivo original:** `CTG.xls`
- **Registros en el archivo de trabajo original:** 2,129
- **Variables originales:** 42

Los datos contienen características previamente extraídas de registros de cardiotocografía. Por lo tanto, el proyecto no trabaja directamente con las señales clínicas originales.

Después del proceso de limpieza y selección de variables se obtuvo un conjunto de:

- **Registros finales:** 2,126
- **Variables predictoras:** 21
- **Variable objetivo:** NSP
- **Valores nulos:** 0

### Distribución de clases

| Clase | Registros | Porcentaje |
|---|---:|---:|
| Normal | 1,655 | 77.85% |
| Sospechoso | 295 | 13.88% |
| Patológico | 176 | 8.28% |

La distribución muestra un desequilibrio importante: la mayoría de los registros pertenecen a la categoría normal, mientras que los casos patológicos representan una proporción menor.

Esto es relevante porque un modelo podría obtener una exactitud general elevada y, aun así, presentar dificultades para identificar los casos menos frecuentes. Por esa razón, el análisis no se limita a Accuracy y considera también métricas como F1 macro y Recall por clase.

## Preparación de los datos

El procesamiento contempló las siguientes actividades:

1. Revisión de la estructura y calidad del conjunto original.
2. Eliminación de los tres registros que no contaban con una etiqueta NSP válida.
3. Eliminación de columnas de metadatos y variables auxiliares o intermedias de clasificación.
4. Selección de 21 características predictoras.
5. Separación de las variables predictoras y la variable objetivo.
6. División estratificada en entrenamiento y prueba.

No se realizó imputación de valores faltantes ni eliminación automática de valores atípicos.

Para los modelos sensibles a la escala de las variables se aplicó estandarización ajustada únicamente con los datos de entrenamiento.

## Modelos evaluados

Se implementaron y compararon cuatro algoritmos:

- Random Forest
- Gradient Boosting
- Regresión Logística One-vs-Rest (OvR)
- K-Nearest Neighbors (KNN)

Cada algoritmo cuenta con una implementación propia y una versión equivalente utilizando Scikit-learn.

## Configuración experimental

Los modelos se evaluaron mediante una división estratificada:

- **Entrenamiento:** 80% (1,701 registros).
- **Prueba:** 20% (425 registros).
- **Semilla aleatoria:** 42.

Se utilizaron las siguientes métricas:

- Accuracy.
- F1-score macro.
- AUC macro.
- Recall por clase.

El uso de métricas macro permite considerar el desempeño en las tres categorías sin que la clase mayoritaria domine por completo la evaluación.

## Resultados y hallazgos

| Modelo | Implementación | Accuracy | F1 macro | AUC macro |
|---|---|---:|---:|---:|
| Random Forest | Propia | 91.53% | 87.02% | 0.9758 |
| Random Forest | Scikit-learn | 92.71% | 88.36% | 0.9792 |
| Gradient Boosting | Propia | 90.35% | 85.38% | 0.9560 |
| Gradient Boosting | Scikit-learn | 92.24% | 87.65% | 0.9762 |
| Regresión Logística OvR | Propia | 87.76% | 76.72% | 0.9518 |
| Regresión Logística OvR | Scikit-learn | 87.29% | 76.59% | 0.9569 |
| KNN | Propia | 90.12% | 81.64% | 0.9460 |
| KNN | Scikit-learn | 89.88% | 80.77% | 0.9454 |

### ¿Qué significan estos resultados para el problema?

**1. Se logró una clasificación general elevada.**

Random Forest de Scikit-learn obtuvo el mejor resultado global, con 92.71% de Accuracy y 88.36% de F1 macro. Esto muestra que las características disponibles contienen información útil para distinguir los tres estados de referencia.

**2. Los modelos propios también lograron identificar una proporción elevada de casos patológicos.**

Random Forest y Gradient Boosting, en sus implementaciones propias, alcanzaron un Recall de 94.29% para la clase patológica. En el conjunto de prueba identificaron correctamente 33 de los 35 registros de esta categoría.

**3. La clase sospechosa sigue siendo un reto relevante.**

La clasificación de registros sospechosos es importante porque representa una categoría intermedia. Su identificación puede resultar más compleja que distinguir registros claramente normales o patológicos. En nuestro experimento, Gradient Boosting propio obtuvo un Recall de 84.75% para esta categoría.

**4. La elección del modelo depende de lo que se quiera priorizar.**

El modelo con mejor desempeño global no necesariamente es el que obtiene el mayor Recall patológico. Por eso, en un problema relacionado con la salud, la evaluación debe considerar los errores por categoría y no únicamente el porcentaje total de aciertos.

### Aportación del proyecto

El resultado principal es la construcción de un proceso **experimental** que permite comparar diferentes enfoques de clasificación sobre características cardiotocográficas y analizar sus aciertos y errores desde la perspectiva del problema.

Los resultados muestran que es posible reconocer una proporción importante de los registros patológicos presentes en el conjunto de prueba. También evidencian que todavía existen casos que pueden clasificarse incorrectamente, por lo que sería necesario realizar más investigación y validación antes de considerar una aplicación clínica.

## Limitaciones

- El conjunto de datos presenta un desequilibrio entre clases.
- Los registros contienen características previamente extraídas, no señales cardiotocográficas originales.
- Los resultados corresponden a una evaluación experimental con una división específica de entrenamiento y prueba.
- No se ha realizado validación clínica ni evaluación en datos hospitalarios externos.
- Los modelos no han sido calibrados para producir probabilidades clínicas.
- Las predicciones no sustituyen la interpretación de especialistas ni los protocolos de atención obstétrica.

Por estas razones, el proyecto debe entenderse como una investigación académica sobre clasificación de datos cardiotocográficos, no como una herramienta de diagnóstico o tratamiento.

## Estructura del repositorio

```text
fetal-health-classifier-ml/
│
├── data/
│   ├── raw/
│   │   ├── CTG.xls
│   │   └── README.md
│   │
│   └── processed/
│       ├── CTG_clean.csv
│       ├── X_features.csv
│       └── y_target.csv
│
├── notebooks/
│   ├── 01_EDA_Limpieza.ipynb
│   ├── 02_EDA_Visualizacion.ipynb
│   ├── 03_Aplicacion_RandomForest.ipynb
│   ├── 04_Aplicacion_GradientBoosting.ipynb
│   ├── 05_Aplicacion_LogisticRegression_OvR.ipynb
│   ├── 06_Aplicacion_Knn.ipynb
│   └── 07_Comparacion_Modelos.ipynb
│
├── src/
│   ├── DecisionTreeClassifier.py
│   ├── RegressionTree.py
│   ├── RandomForest.py
│   ├── GradientBoosting.py
│   ├── Knn.py
│   ├── LogisticRegression.py
│   └── utils_and_metrics.py
│
├── docs/
│   ├── README.md
│   ├── Poster.pdf
│   ├── ReporteFinal/
│   └── ReporteComplejidad/
│
├── requirements.txt
└── README.md
```

## Instalación y ejecución

### Requisitos

- Python 3.10 o superior.
- pip.
- JupyterLab.

### 1. Clonar el repositorio

```bash
git clone https://github.com/DylanRico007/fetal-health-classifier-ml.git
cd fetal-health-classifier-ml
```

### 2. Crear un entorno virtual

```bash
python -m venv venv
```

Activar en Windows:

```bash
venv\Scripts\activate
```

Activar en macOS o Linux:

```bash
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Abrir JupyterLab

```bash
cd notebooks
jupyter lab
```

Se recomienda ejecutar los notebooks en orden numérico, comenzando por el análisis y limpieza de los datos.

El notebook `07_Comparacion_Modelos.ipynb` presenta la comparación de resultados previamente obtenidos y no vuelve a entrenar los modelos.

## Documentación

En la carpeta `docs/` se encuentran los entregables del proyecto:

- **Poster:** resumen visual del problema, metodología y resultados.
- **Reporte final de Machine Learning:** documentación del desarrollo, fundamentos, modelos y resultados.
- **Reporte de complejidad computacional:** análisis teórico y experimental de KNN y Regresión Logística OvR.

Cada documento responde a los objetivos de su respectiva asignatura, por lo que su alcance y selección de algoritmos pueden ser diferentes.

## Referencias

- UCI Machine Learning Repository. (2010). *Cardiotocography*. https://doi.org/10.24432/C51S4N
- Ayres-de-Campos, D., Bernardes, J., Garrido, A., Marques-de-Sá, J. P., & Pereira-Leite, L. (2000). SisPorto 2.0: A program for automated analysis of cardiotocograms. *Journal of Maternal-Fetal Medicine, 9*(5), 311–318.

## Tecnologías utilizadas

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy
- JupyterLab
