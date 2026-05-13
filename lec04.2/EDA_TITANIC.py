import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# CARGA DEL DATASET
df = pd.read_csv("Titanic-Dataset.csv")
print(df.head())

# SELECCIÓN DE VARIABLES NUMÉRICAS
df_num = df.select_dtypes(include=np.number)
print(df_num.head())

# MEDIDAS DE TENDENCIA CENTRAL
print("Media:")
print(df_num.mean())

print("\nMediana:")
print(df_num.median())

print("\nModa:")
print(df_num.mode().iloc[0])

# MEDIDAS DE DISPERSIÓN
print("\nDesviación estándar:")
print(df_num.std())

print("\nVarianza:")
print(df_num.var())

print("\nMínimo:")
print(df_num.min())

print("\nMáximo:")
print(df_num.max())

# MEDIDAS DE POSICIÓN
print("\nCuartiles:")
print(df_num.quantile([0.25, 0.5, 0.75]))

print("\nPercentil 90:")
print(df_num.quantile(0.90))

# DETECCIÓN DE OUTLIERS (IQR)
Q1 = df_num.quantile(0.25)
Q3 = df_num.quantile(0.75)
IQR = Q3 - Q1

lim_inf = Q1 - 1.5 * IQR
lim_sup = Q3 + 1.5 * IQR

outliers = (df_num < lim_inf) | (df_num > lim_sup)

print("\nCantidad de outliers por columna:")
print(outliers.sum())

# ELIMINACIÓN DE OUTLIERS (EXPLORATORIA)
df_sin_outliers = df_num[
    ~((df_num < lim_inf) | (df_num > lim_sup)).any(axis=1)
]

print("\nDimensiones originales:", df_num.shape)
print("Dimensiones sin outliers:", df_sin_outliers.shape)

# HISTOGRAMAS
df_num.hist(bins=20, figsize=(12, 8))
plt.suptitle("Histogramas de variables numéricas")
plt.tight_layout()
plt.show()

# GRÁFICO DE DISPERSIÓN
x = df_num.columns[0]
y = df_num.columns[1]

plt.figure(figsize=(8, 6))
plt.scatter(df_num[x], df_num[y])
plt.xlabel(x)
plt.ylabel(y)
plt.title("Gráfico de dispersión entre dos variables")
plt.show()

# MATRIZ DE CORRELACIÓN
corr = df_num.corr()
print("\nMatriz de correlación:")
print(corr)

plt.figure(figsize=(10, 8))
plt.imshow(corr)
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Matriz de correlación")
plt.show()

# ONE HOT ENCODING
df_onehot = pd.get_dummies(df, drop_first=True)
print(df_onehot.head())

# LABEL ENCODING
from sklearn.preprocessing import LabelEncoder

df_label = df.copy()
le = LabelEncoder()

for col in df_label.select_dtypes(exclude=np.number).columns:
    df_label[col] = le.fit_transform(df_label[col].astype(str))

print(df_label.head())

# BINARY ENCODING
import category_encoders as ce

df_binary = df.copy()
binary_encoder = ce.BinaryEncoder(
    cols=df_binary.select_dtypes(exclude=np.number).columns
)

df_binary = binary_encoder.fit_transform(df_binary)
print(df_binary.head())

# ESCALAMIENTO DE VARIABLES
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Min-Max Scaling
minmax = MinMaxScaler()
df_minmax = pd.DataFrame(
    minmax.fit_transform(df_num),
    columns=df_num.columns
)

print(df_minmax.head())

# Standard Scaler
standard = StandardScaler()
df_standard = pd.DataFrame(
    standard.fit_transform(df_num),
    columns=df_num.columns
)

print(df_standard.head())

# TRANSFORMACIÓN LOGARÍTMICA
df_log = df_num.copy()

for col in df_log.columns:
    if (df_log[col] > 0).all():
        df_log[col] = np.log(df_log[col])

print(df_log.head())

# ===============================
# CONCLUSIONES
# ===============================
# - Se observa que varias variables, especialmente Fare, presentan distribuciones sesgadas y no normales, evidenciado en los histogramas.
#
# - Los valores atípicos aparecen principalmente en Fare, SibSp y Parch, y están asociados a pasajeros con tarifas altas o familias más grandes.
#
# - Com la correlación entre Fare y Pclass podemos llegar a que el nivel socioeconómico influyó en las probabilidades de sobrevivir.
#
# - El análisis estadístico permitió entender mejor el comportamiento de los datos antes de aplicar modelos, resaltando la importancia de explorar la información previamente.