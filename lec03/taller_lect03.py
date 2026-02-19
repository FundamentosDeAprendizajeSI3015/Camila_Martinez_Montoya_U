import pandas as pd
import numpy as np

# ---------------------------
# 1. CARGA DEL CONJUNTO DE DATOS
# ---------------------------
df = pd.read_csv('Titanic-Dataset.csv')

print("=== 1. Carga del conjunto de datos ===")
print(f"Filas: {len(df)}, Columnas: {len(df.columns)}")
print(df)

# ---------------------------
# 2. INSPECCIÓN RÁPIDA
# ---------------------------
print("\n=== 2. Inspección rápida ===")

print("\nPrimeras 5 filas:")
print(df.head())

print("\nInfo (tipos y no nulos):")
print(df.info())

print("\nEstadísticas descriptivas:")
print(df.describe(include='all'))

print("\nValores únicos por columna:")
print(df.nunique())

# ---------------------------
# 3. MANEJO DE VALORES FALTANTES (NaNs)
# ---------------------------
print("\n=== 3. Manejo de valores faltantes ===")

print("NaNs por columna (antes):")
print(df.isna().sum())

# Imputación de valores faltantes
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Cabin'] = df['Cabin'].fillna('SIN_CABINA')
df['Embarked'] = df['Embarked'].fillna('S')
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

print("\nNaNs por columna (después):")
print(df.isna().sum())
print("Verificación: ¿Quedan NaNs?", df.isna().any().any())

# ---------------------------
# 4. MANIPULACIÓN DE FILAS Y COLUMNAS
# ---------------------------
print("\n=== 4. Manipulación de filas y columnas ===")

df = df.drop(columns=['Ticket'], errors='ignore')
df = df[df['Fare'] > 0].copy()

# Reordenar columnas: Survived al final
cols = [c for c in df.columns if c != 'Survived'] + ['Survived']
df = df[cols]

print("Columnas tras manipulación:", list(df.columns))
print("Filas tras filtrar Fare > 0:", len(df))

# ---------------------------
# 5. LIMPIEZA DE TEXTO Y MANEJO DE DUPLICADOS
# ---------------------------
print("\n=== 5. Limpieza de texto y duplicados ===")

df['Name'] = df['Name'].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)

# Extraer deck de la cabina
df['Cabin_Deck'] = np.where(
    df['Cabin'].str.match(r'^[A-Z]\d', na=False),
    df['Cabin'].str[0],
    'X'
)

duplicados = df.duplicated(subset=['PassengerId']).sum()
print(f"Duplicados por PassengerId: {duplicados}")
df = df.drop_duplicates(subset=['PassengerId'], keep='first')
print("Filas tras eliminar duplicados:", len(df))

# ---------------------------
# 6. CONSISTENCIA Y VALIDACIÓN LÓGICA
# ---------------------------
print("\n=== 6. Consistencia y validación lógica ===")

df = df[(df['Age'] >= 0) & (df['Age'] <= 100)]
df = df[df['Pclass'].isin([1, 2, 3])]
df = df[df['Survived'].isin([0, 1])]

print("Filas tras validación lógica:", len(df))

# ---------------------------
# 7. TRANSFORMACIÓN DE TIPOS Y FILTRADO
# ---------------------------
print("\n=== 7. Transformación de tipos ===")

df['Pclass']   = df['Pclass'].astype('category')
df['Sex']      = df['Sex'].astype('category')
df['Embarked'] = df['Embarked'].astype('category')

df['Age']  = pd.to_numeric(df['Age'],  errors='coerce')
df['Fare'] = pd.to_numeric(df['Fare'], errors='coerce')

# Filtrar outliers de Fare (percentil 99)
q99 = df['Fare'].quantile(0.99)
df = df[df['Fare'] <= q99].copy()

print(f"Fare máximo tras filtrar outliers (q99): {df['Fare'].max():.2f}")

# ---------------------------
# 8. OPERACIONES DE AGREGACIÓN Y AGRUPAMIENTO
# ---------------------------
print("\n=== 8. Agregación y agrupamiento ===")

agg_clase = df.groupby('Pclass').agg(
    supervivientes=('Survived', 'sum'),
    total=('Survived', 'count'),
    edad_promedio=('Age', 'mean'),
    fare_promedio=('Fare', 'mean')
).assign(tasa_supervivencia=lambda x: x['supervivientes'] / x['total'])

print("Agregación por Pclass:")
print(agg_clase)

agg_sex = (
    df.groupby('Sex')['Survived']
    .agg(['sum', 'count', 'mean'])
    .rename(columns={'mean': 'tasa'})
)
print("\nAgregación por Sex:")
print(agg_sex)

# ---------------------------
# 9. TRANSFORMACIÓN ONE HOT ENCODING
# ---------------------------
print("\n=== 9. One Hot Encoding ===")

df_encoded = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=False)

print("Columnas generadas por One Hot (Sex, Embarked):")
print([c for c in df_encoded.columns if c.startswith('Sex_') or c.startswith('Embarked_')])

print("\nMuestra de filas codificadas:")
print(df_encoded[['Pclass', 'Age', 'Fare',
                   'Sex_female', 'Sex_male',
                   'Embarked_C', 'Embarked_Q', 'Embarked_S']].head())

# ---------------------------
# RESUMEN FINAL
# ---------------------------
print("\n=== Resumen final ===")
print(f"Dataset final: {df_encoded.shape[0]} filas, {df_encoded.shape[1]} columnas")
print("Limpieza aplicada: NaNs imputados, duplicados eliminados, validación lógica, One Hot Encoding.")
print("\n[OK] Taller completado.")