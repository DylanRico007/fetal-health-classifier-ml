# Datos originales

## Cardiotocography Data Set

Este directorio contiene el conjunto de datos original utilizado en el proyecto de clasificación del estado de salud fetal.

### Relevancia del conjunto de datos

La cardiotocografía permite registrar la frecuencia cardiaca fetal y la actividad uterina, dos elementos utilizados durante la vigilancia obstétrica. El análisis de sus características permite estudiar patrones relacionados con el bienestar fetal.

El conjunto utilizado en este proyecto permite investigar cómo se pueden clasificar esos patrones en tres categorías: normal, sospechoso y patológico. Las etiquetas de referencia fueron asignadas mediante el consenso de tres especialistas en obstetricia.

Es importante señalar que el conjunto contiene características previamente extraídas de registros cardiotocográficos, no señales clínicas originales sin procesar.

### Fuente

- **Nombre:** Cardiotocography Data Set.
- **Repositorio de distribución:** UCI Machine Learning Repository.
- **Archivo:** `CTG.xls`.

El conjunto procede de investigación asociada con Porto, Portugal, y el trabajo de desarrollo de SisPorto. Su distribución pública se realiza a través del repositorio UCI.

### Dimensiones del archivo de trabajo original

| Característica | Valor |
|---|---:|
| Registros | 2,129 |
| Columnas | 42 |
| Archivo | CTG.xls |

### Clasificación de referencia

| NSP | Clasificación |
|---|---|
| 1 | Normal |
| 2 | Sospechoso |
| 3 | Patológico |

## Procesamiento realizado en el proyecto

A partir del archivo original se realizó un proceso adicional de revisión, limpieza y selección de variables.

Las principales modificaciones fueron:

1. Eliminación de tres registros que no contaban con una etiqueta NSP válida.
2. Eliminación de columnas de metadatos y variables auxiliares.
3. Exclusión de variables intermedias relacionadas con la clasificación.
4. Selección de 21 características predictoras y la variable objetivo NSP.

No se realizó imputación de valores faltantes ni se fabricaron datos para modificar la distribución del conjunto.

### Resultado del procesamiento

El conjunto final contiene:

- 2,126 registros.
- 21 variables predictoras.
- Una variable objetivo: NSP.
- Ningún valor nulo.

Los archivos resultantes se encuentran en `data/processed/`.

## Archivos

```text
data/
│
├── raw/
│   ├── CTG.xls
│   └── README.md
│
└── processed/
    ├── CTG_clean.csv
    ├── X_features.csv
    └── y_target.csv
```
