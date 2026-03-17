import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
from sklearn.cluster import DBSCAN, KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    PolynomialFeatures,
    FunctionTransformer,
)

random_state = 42

plt.rc('font', family='serif', size=12)

# Dataset 1
df1 = pd.read_csv("dataset_limpio.csv")
 
# Seleccionamos solo columnas numéricas (excluimos 'anio', 'unidad', 'label')
cols_num_1 = [c for c in df1.columns if c not in ('anio', 'unidad', 'label')]
data = df1[cols_num_1].values
 
fig, ax = plt.subplots()
ax.scatter(data[:, 0], data[:, 1])
fig.set_size_inches(5 * 1.6, 5)
plt.show()
 
# Definir el pipeline de pre-procesamiento
numeric_transformer = Pipeline(
    steps=[("scaler", StandardScaler())]
)
 
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, np.arange(data.shape[1])),
    ],
)
 
np.arange(data.shape[1])
 
# Definimos el Pipeline de clustering con K = 2
clu_kmeans = Pipeline(steps=[("preprocessor", preprocessor), ("clustering", KMeans(n_clusters=2))])
 
# Entrenamos
clu_kmeans.fit(data)
 
fig, ax = plt.subplots()
ax.scatter(data[:, 0], data[:, 1], c=clu_kmeans['clustering'].labels_)
fig.set_size_inches(5 * 1.6, 5)
plt.show()
 
inert = []
k_range = list(range(1, 11))
for k in k_range:
    clu_kmeans = Pipeline(steps=[("preprocessor", preprocessor), ("clustering", KMeans(n_clusters=k))])
    clu_kmeans.fit(data)
    inert.append(clu_kmeans['clustering'].inertia_)
 
fig, ax = plt.subplots()
ax.plot(k_range, inert)
fig.set_size_inches(5 * 1.6, 5)
plt.show()

K_OPTIMO_1 = 3
 
clu_kmeans = Pipeline(steps=[("preprocessor", preprocessor), ("clustering", KMeans(n_clusters=K_OPTIMO_1))])
clu_kmeans.fit(data)
print(f'con K = {K_OPTIMO_1}: la inercia es {clu_kmeans["clustering"].inertia_}')
 
fig, ax = plt.subplots()
ax.scatter(data[:, 0], data[:, 1], c=clu_kmeans['clustering'].labels_)
fig.set_size_inches(5 * 1.6, 5)
plt.show()

#----------------------------------------------------------------------------------------------------------------------------------

# Dataset 2 
df2 = pd.read_csv("dataset_limpio_sintetico.csv")
 
# Seleccionamos solo columnas numéricas (excluimos 'label')
cols_num_2 = [c for c in df2.columns if c != 'label']
data = df2[cols_num_2].values
 
fig, ax = plt.subplots()
ax.scatter(data[:, 0], data[:, 1])
fig.set_size_inches(5 * 1.6, 5)
plt.show()
 
# Definir el pipeline de pre-procesamiento
numeric_transformer = Pipeline(
    steps=[("scaler", StandardScaler())]
)
 
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, np.arange(data.shape[1])),
    ],
)
 
# Definimos el Pipeline de clustering con K = 2
clu_kmeans = Pipeline(steps=[("preprocessor", preprocessor), ("clustering", KMeans(n_clusters=2))])
clu_kmeans.fit(data)
print(f'con K = 2: la inercia es {clu_kmeans["clustering"].inertia_}')
 
fig, ax = plt.subplots()
ax.scatter(data[:, 0], data[:, 1], c=clu_kmeans['clustering'].labels_)
fig.set_size_inches(5 * 1.6, 5)
plt.show()
 
inert = []
k_range = list(range(1, 11))
for k in k_range:
    clu_kmeans = Pipeline(steps=[("preprocessor", preprocessor), ("clustering", KMeans(n_clusters=k))])
    clu_kmeans.fit(data)
    inert.append(clu_kmeans['clustering'].inertia_)
 
fig, ax = plt.subplots()
ax.plot(k_range, inert)
fig.set_size_inches(5 * 1.6, 5)
plt.show()

K_OPTIMO_2 = 5
 
clu_kmeans = Pipeline(steps=[("preprocessor", preprocessor), ("clustering", KMeans(n_clusters=K_OPTIMO_2))])
clu_kmeans.fit(data)
print(f'con K = {K_OPTIMO_2}: la inercia es {clu_kmeans["clustering"].inertia_}')
 
fig, ax = plt.subplots()
ax.scatter(data[:, 0], data[:, 1], c=clu_kmeans['clustering'].labels_)
fig.set_size_inches(5 * 1.6, 5)
plt.show()
 
clu_dbscan = Pipeline(steps=[("clustering", DBSCAN(eps=0.1, min_samples=10))])
clu_dbscan.fit(data)
 
fig, ax = plt.subplots()
ax.scatter(data[:, 0], data[:, 1], c=clu_dbscan['clustering'].labels_)
fig.set_size_inches(5 * 1.6, 5)
plt.show()
 
np.unique(clu_dbscan['clustering'].labels_, return_counts=True)