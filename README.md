# Datos originales: Cardiotocografía (Fetal Health)

## Procedencia de los datos

- **Conjunto de datos:** Cardiotocography (CTG).
- **Procedencia original:** Biomedical Engineering Institute y Faculty of Medicine, University of Porto, Porto, Portugal.
- **Autores asociados:** J. P. Marques de Sá, J. Bernardes y D. Ayres de Campos.
- **Repositorio de distribución:** UCI Machine Learning Repository.
- **Archivo utilizado:** `CTG.xls`.

Los registros corresponden a cardiotocografías que fueron procesadas automáticamente mediante herramientas de análisis como SisPorto 2.0, de las cuales se obtuvieron características diagnósticas. Posteriormente, tres obstetras clasificaron los registros mediante un criterio de consenso. Por tanto, el proyecto utiliza un conjunto de datos previamente procesado, sobre el cual se realizó una etapa adicional de limpieza, selección de características y preparación para los modelos de Machine Learning.

## Descripción

Este directorio contiene el archivo original utilizado como fuente de datos para el proyecto. Incluye registros de cardiotocografía y sus respectivas variables clínicas, así como la etiqueta objetivo `NSP`.

El archivo original se conserva sin modificaciones. Su inspección, limpieza y transformación se realizan en el notebook `01_EDA_Limpieza.ipynb`.

## Procesamiento aplicado

Durante la preparación de los datos:

1. Se eliminaron tres registros que no contenían una etiqueta objetivo `NSP` válida.
2. Se descartaron metadatos temporales y variables auxiliares que no se utilizaron como predictores.
3. Se conservaron las tasas normalizadas de los eventos cardiotocográficos.
4. Se excluyeron las variables de clasificación intermedia y `CLASS` para evitar posibles fugas de información.
5. Se obtuvo un conjunto procesado de 2,126 registros, 21 características predictoras y la variable objetivo `NSP`.

No se realizó imputación artificial de valores faltantes.

Los archivos procesados se almacenan en `data/processed/`.

