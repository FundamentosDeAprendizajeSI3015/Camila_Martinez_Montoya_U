Informe – Análisis Exploratorio de Datos (EDA) para Riesgo Operativo

1. Introducción

En este trabajo realizamos un análisis exploratorio de datos (EDA) sobre un dataset de riesgo operativo.

La idea principal es entender bien la información que tenemos antes de pasar a la parte de modelos de Machine Learning. En esta fase del proyecto solo nos enfocamos en la exploración de los datos:

- Carga de datos
- Inspección inicial del dataset
- Cálculo de estadísticas descriptivas
- Visualización mediante histogramas
- Detección de valores atípicos (outliers) usando el método IQR

En esta etapa todavía no se entrena ningún modelo ni se hace evaluación.

2. Descripción del dataset

El dataset que usamos se encuentra en el archivo:

data/raw/dataset_riesgo_operativo.csv

Cada fila representa una observación relacionada con el desempeño de un equipo en determinadas semanas. A nivel general, incluye variables como:

- equipo
- semana
- t_prom
- pct_fuera
- pct_cumplimiento
- interaccion
- des_carga
- correcciones
- riesgo_final

La mayoría de las variables son numéricas (tiempos, porcentajes, conteos), lo que nos permite analizar cómo se comporta el riesgo operativo bajo distintas condiciones y empezar a sacar conclusiones básicas sobre el comportamiento de los equipos.

3. Estructura del proyecto

El proyecto está organizado con la siguiente estructura principal:

- main.py: ejecuta todo el pipeline de EDA.

- cargar_data.py: contiene funciones para cargar el dataset y mostrar una muestra inicial.

- eda.py: contiene las funciones de análisis exploratorio.

Esta organización modular ayuda a entender qué hace cada parte del código y mantiene el proyecto más ordenado y fácil de revisar.

4. Metodología y pipeline de EDA

El pipeline completo se ejecuta desde la función `main()` en `main.py`. La idea es que siempre se ejecute en el mismo orden.

4.1 Carga de datos

En 'cargar_data.py' se implementan dos funciones principales:

- 'load_data(path)':  
  Verifica que el archivo exista y lo carga usando `pandas.read_csv`.  
  También muestra cuántas filas y columnas se cargaron.

- 'mostrar_muestra(df, n=5)':  
  Imprime las primeras filas del dataset para revisar rápidamente que los datos se hayan cargado bien.

En 'main.py' se define la ruta del archivo:

'RUTA_DATASET = "../data/raw/dataset_riesgo_operativo.csv"'

4.2 Información básica del dataset

En la función 'informacion_basica(df)' se obtiene una visión general del dataset:

- Número de filas y columnas.
- Tipos de datos por columna.
- Cantidad y porcentaje de valores nulos..

4.3 Estadísticas descriptivas

En 'estadisticas_descriptivas(df)' se calculan estadísticas básicas para todas las columnas numéricas:

- Media
- Mediana
- Moda
- Desviación estándar
- Mínimo y máximo

Estas métricas permiten entender el rango de los datos, qué tan dispersos están y si hay valores que se salen mucho de lo normal.

4.4 Visualización con histogramas

En 'graficar_histogramas(df)' se genera un histograma por cada variable numérica para observar su distribución.

Se utiliza 'matplotlib' y se dibujan líneas verticales que indican la media y la mediana de cada variable, lo que ayuda a ver si la distribución está muy cargada hacia un lado o si es más simétrica.

Los gráficos se guardan como archivos '.png' en: outputs/graficos/

4.5 Detección de outliers con el método IQR

En 'detectar_outliers(df)' se utiliza el método del rango intercuartílico (IQR) para detectar valores atípicos.

Para cada columna numérica:

- Se calcula Q1 (percentil 25)
- Se calcula Q3 (percentil 75)
- Se obtiene el IQR como Q3 − Q1
- Se definen límites inferior y superior
- Los valores fuera de esos límites se consideran outliers.

La función recorre todas las columnas numéricas y muestra cuántos outliers se detectaron por variable y en total. Esto nos da una idea de qué variables podrían necesitar un tratamiento especial más adelante.

5. Resultados generales del EDA

Al finalizar la ejecución del pipeline, se muestra un resumen con:

- Ruta del dataset analizado
- Número de filas y columnas
- Cantidad de variables numéricas
- Número total de valores nulos
- Número total de outliers detectados
- Columnas donde se encontraron outliers
- Ruta donde se guardan los gráficos

6. Conclusiones

A partir de este análisis exploratorio se logró:

- Entender la estructura general del dataset.
- Analizar el comportamiento de las variables numéricas.
- Detectar valores nulos y posibles outliers.
- Generar visualizaciones que facilitan la interpretación de los datos.

