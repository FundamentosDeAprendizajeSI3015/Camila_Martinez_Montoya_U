
import pandas as pd
from numpy import nan
import numpy as np

# Cargar el dataset FIRE UdeA
df = pd.read_csv('dataset_sintetico_FIRE_UdeA_realista.csv')
df

# Ver las primeras 5 filas
df.head()

# Ver las últimas 5 filas
df.tail()

# Dimensiones (filas, columnas)
df.shape

# Información general: tipos de datos y valores no nulos por columna
df.info()

# Tipos de datos de cada columna
df.dtypes

# Estadísticas descriptivas (media, min, max, percentiles)
df.describe()

# Conteo de valores únicos por columna
df.nunique()

# Conteo de valores únicos de la columna categórica 'unidad'
df['unidad'].nunique()

# Valores únicos en 'unidad'
df['unidad'].unique()


# Detectar nulos por columna (suma total)
df.isnull().sum()

# Total de valores nulos en todo el DataFrame
df.isnull().sum().sum()

# Porcentaje de NaN por columna
(df.isnull().sum() / len(df) * 100).round(2)

# Visualizar SOLO las columnas con al menos un nulo
nulos = df.isnull().sum()
print("Columnas con nulos:\n")
print(nulos[nulos > 0])

# Opción 1 — Eliminar filas que tengan AL MENOS un nulo
df_sin_nulos = df.dropna()
print(f"Filas originales: {len(df)}  →  Filas sin nulos: {len(df_sin_nulos)}")
df_sin_nulos.head()

# Opción 2 — Eliminar columnas que tengan AL MENOS un nulo
df_cols_completas = df.dropna(axis=1)
print(f"Columnas originales: {df.shape[1]}  →  Columnas sin nulos: {df_cols_completas.shape[1]}")
df_cols_completas.head()

# Opción 3 — Conservar filas con al menos N valores no nulos (thresh)
df_thresh = df.dropna(thresh=14)   # mínimo 14 de 16 columnas no nulas
print(f"Filas conservadas con thresh=14: {len(df_thresh)}")

# Opción 4 — Eliminar filas donde TODAS las columnas son nulas
df_all = df.dropna(how='all')
print(f"Filas eliminadas (todo nulo): {len(df) - len(df_all)}")

# Opción 5 — Rellenar nulos con un valor constante (0)
df_cero = df.copy()
df_cero.fillna(0, inplace=True)
print("Nulos tras rellenar con 0:", df_cero.isnull().sum().sum())

# Opción 6 — Rellenar nulos en columna específica con su media global
df_media = df.copy()
df_media.fillna({'endeudamiento': df['endeudamiento'].mean()}, inplace=True)
print(f"Media de endeudamiento: {df['endeudamiento'].mean():.4f}")
df_media[['unidad','anio','endeudamiento']].head(10)

# Opción 7 — Imputación inteligente: mediana agrupada por unidad
# Esta es la estrategia más adecuada para datos financieros por facultad.
df_imputado = df.copy()
cols_con_nulos = df.columns[df.isnull().any()].difference(['anio','label'])

for col in cols_con_nulos:
    df_imputado[col] = df_imputado.groupby('unidad')[col].transform(
        lambda x: x.fillna(x.median())
    )
# Fallback global si algún grupo quedó completamente nulo
df_imputado[cols_con_nulos] = df_imputado[cols_con_nulos].fillna(df[cols_con_nulos].median())

print("Nulos tras imputación por mediana de grupo:", df_imputado.isnull().sum().sum())
df_imputado.head()

# Opción 8 — bfill (rellenar con el valor siguiente en la columna)
df_bfill = df.copy()
df_bfill['cfo'] = df_bfill['cfo'].bfill()
print("Nulos en 'cfo' tras bfill:", df_bfill['cfo'].isnull().sum())

# Opción 9 — ffill (rellenar con el valor anterior en la columna)
df_ffill = df.copy()
df_ffill['cfo'] = df_ffill['cfo'].ffill()
print("Nulos en 'cfo' tras ffill:", df_ffill['cfo'].isnull().sum())

# Opción 10 — Interpolación lineal (útil para series de tiempo)
df_interp = df.copy()
df_interp['cfo'] = df_interp['cfo'].interpolate()
print("Nulos en 'cfo' tras interpolación:", df_interp['cfo'].isnull().sum())

# Para variables categóricas, rellenar con 'Desconocido' (no aplica aquí, demostración)
# En FIRE UdeA 'unidad' no tiene nulos, pero si los tuviera:
df_cat = df.copy()
df_cat['unidad'] = df_cat['unidad'].fillna('Desconocido')
print("Nulos en 'unidad':", df_cat['unidad'].isnull().sum())


# Cargamos de nuevo el archivo original para el pipeline completo
df_raw = pd.read_csv('dataset_sintetico_FIRE_UdeA_realista.csv')

print("=== DICCIONARIO DE VARIABLES ===")
diccionario = {
    'anio':                   'Año del registro (2016-2025)',
    'unidad':                 'Unidad académica (facultad, sede, instituto)',
    'ingresos_totales':       'Ingresos totales en COP',
    'gastos_personal':        'Gastos en personal (nómina)',
    'liquidez':               'Razón corriente (activo/pasivo corriente)',
    'dias_efectivo':          'Días de efectivo disponibles',
    'cfo':                    'Flujo de caja operacional (Cash Flow from Operations)',
    'participacion_ley30':    'Participación de recursos Ley 30 en ingresos',
    'participacion_regalias': 'Participación de regalías en ingresos',
    'participacion_servicios':'Participación de servicios académicos en ingresos',
    'participacion_matriculas':'Participación de matrículas en ingresos',
    'hhi_fuentes':            'Índice Herfindahl-Hirschman de diversificación de fuentes',
    'endeudamiento':          'Razón de endeudamiento (pasivo/activo)',
    'tendencia_ingresos':     'Tasa de cambio interanual de ingresos',
    'gp_ratio':               'Razón gastos de personal / ingresos totales',
    'label':                  'Variable objetivo: 0=Bajo Riesgo, 1=Alto Riesgo',
}
for col, desc in diccionario.items():
    print(f"  {col:<28} → {desc}")

# Paso 1 — Inspección inicial
print(f"Shape: {df_raw.shape}")
print(f"Duplicados: {df_raw.duplicated().sum()}")
print()
print("Nulos por columna:")
print(df_raw.isnull().sum()[df_raw.isnull().sum() > 0])

# Paso 2 — Eliminar duplicados (no hay en este caso, pero es buena práctica)
df_clean = df_raw.drop_duplicates()
print(f"Filas tras eliminar duplicados: {len(df_clean)}")

# Paso 3 — Verificar consistencia lógica
# gp_ratio debe ser gastos_personal / ingresos_totales
df_clean['gp_ratio_check'] = df_clean['gastos_personal'] / df_clean['ingresos_totales']
diferencia = (df_clean['gp_ratio'] - df_clean['gp_ratio_check']).abs().max()
print(f"Diferencia máxima en gp_ratio (validación): {diferencia:.6f}")
df_clean.drop(columns='gp_ratio_check', inplace=True)

# Paso 4 — Validación de rangos lógicos
print("Valores de liquidez < 0:", (df_clean['liquidez'] < 0).sum())
print("Valores de endeudamiento fuera de [0,1]:", ((df_clean['endeudamiento'] < 0) | (df_clean['endeudamiento'] > 1)).sum())
print("Valores de participaciones fuera de [0,1]:")
for col in ['participacion_ley30','participacion_regalias','participacion_servicios','participacion_matriculas']:
    fuera = ((df_clean[col] < 0) | (df_clean[col] > 1)).sum()
    print(f"  {col}: {fuera}")

# Paso 5 — Imputación: mediana por unidad para variables numéricas con nulos
cols_nulos = df_clean.columns[df_clean.isnull().any()].difference(['anio','label'])
print("Columnas a imputar:", list(cols_nulos))

for col in cols_nulos:
    df_clean[col] = df_clean.groupby('unidad')[col].transform(
        lambda x: x.fillna(x.median())
    )
# Fallback: mediana global
df_clean[cols_nulos] = df_clean[cols_nulos].fillna(df_clean[cols_nulos].median())

print("\nNulos tras imputación:", df_clean.isnull().sum().sum())

# Paso 6 — Eliminar columnas de varianza cero (no aportan información)
cols_antes = df_clean.shape[1]
df_clean = df_clean.loc[:, df_clean.nunique() > 1]
print(f"Columnas antes: {cols_antes}  →  después: {df_clean.shape[1]}")

# Paso 7 — Conversión de tipos de dato
# 'anio' puede quedar como int, 'label' como int, numéricas como float64
df_clean['anio']  = df_clean['anio'].astype(int)
df_clean['label'] = df_clean['label'].astype(int)
print("Tipos finales:")
print(df_clean.dtypes)

# Paso 8 — Reordenar columnas (identificadores primero, target al final)
orden = ['anio','unidad','ingresos_totales','gastos_personal','liquidez','dias_efectivo',
         'cfo','participacion_ley30','participacion_regalias','participacion_servicios',
         'participacion_matriculas','hhi_fuentes','endeudamiento','tendencia_ingresos',
         'gp_ratio','label']
df_clean = df_clean[orden]
print("Shape final:", df_clean.shape)
df_clean.head(10)

# Paso 9 — Guardar el dataset limpio
df_clean.to_csv('dataset_limpio.csv', index=False)
print("dataset_limpio.csv guardado correctamente.")


# Eliminar columnas específicas (ejemplo: quitar 'hhi_fuentes' y 'tendencia_ingresos')
df_sin_cols = df_clean.drop(['hhi_fuentes', 'tendencia_ingresos'], axis=1)
df_sin_cols.head()

# Eliminar filas por índice (ejemplo: primeras 3 filas)
df_clean.drop([0, 1, 2], axis=0)

# Renombrar columnas (ejemplo: traducción al inglés de algunas columnas)
df_renombrado = df_clean.rename(columns={
    'anio': 'year',
    'unidad': 'unit',
    'ingresos_totales': 'total_revenue',
    'gastos_personal': 'personnel_expenses',
})
df_renombrado.head()

# Reordenar columnas: poner 'label' al inicio para visualizar el target
df_reordenado = df_clean[['label'] + [c for c in df_clean.columns if c != 'label']]
df_reordenado.head()

# Crear una nueva columna: margen operacional (ingresos - gastos) en miles de millones
df_clean['margen_operacional'] = (df_clean['ingresos_totales'] - df_clean['gastos_personal']) / 1e9
df_clean[['anio','unidad','margen_operacional']].head(10)


# Verificar duplicados exactos
print("Filas duplicadas:", df_raw.duplicated().sum())

# Verificar duplicados en el par (anio, unidad) — deben ser únicos
print("Duplicados en (anio, unidad):", df_raw.duplicated(subset=['anio','unidad']).sum())

# Verificar valores únicos de 'unidad' y limpiar espacios/case
print("Valores únicos originales:")
print(df_raw['unidad'].unique())

df_texto = df_raw.copy()
df_texto['unidad'] = df_texto['unidad'].str.strip()   # quitar espacios al inicio/fin
print("\nValores únicos tras limpieza de texto:")
print(df_texto['unidad'].unique())

# Eliminar filas duplicadas (demostración sobre df_raw)
df_sin_dup = df_raw.drop_duplicates()
print(f"Filas antes: {len(df_raw)}  →  después: {len(df_sin_dup)}")


# Filtrado: conservar solo registros con liquidez dentro de un rango razonable [0.5, 3]
df_filtrado = df_clean[(df_clean['liquidez'] >= 0.5) & (df_clean['liquidez'] <= 3.0)]
print(f"Registros dentro del rango de liquidez [0.5, 3]: {len(df_filtrado)} de {len(df_clean)}")

# Eliminar columnas con un único valor (varianza cero)
df_test_var = df_clean.copy()
df_test_var['col_constante'] = 999   # simulamos una columna constante
df_filtrado_var = df_test_var.loc[:, df_test_var.nunique() > 1]
print(f"Columnas eliminadas por varianza cero: {df_test_var.shape[1] - df_filtrado_var.shape[1]}")


# El campo 'anio' ya es int, pero si viniera como float lo convertiríamos:
df_tipos = df_clean.copy()
df_tipos['anio'] = df_tipos['anio'].astype(int)
print("Tipo de 'anio':", df_tipos['anio'].dtype)

# Filtrar registros de un año específico
df_2020 = df_clean[df_clean['anio'] == 2020]
print(f"Registros del año 2020: {len(df_2020)}")
df_2020

# Filtrar por unidad específica
df_medicina = df_clean[df_clean['unidad'] == 'Medicina']
print(f"Registros de Medicina: {len(df_medicina)}")
df_medicina[['anio','ingresos_totales','gp_ratio','label']]


# Promedio de ingresos totales por unidad académica
promedio_ingresos = df_clean.groupby('unidad')['ingresos_totales'].mean() / 1e9
promedio_ingresos = promedio_ingresos.sort_values(ascending=False)
print("Promedio de Ingresos Totales por Unidad (miles de millones COP):")
print(promedio_ingresos.round(2))

# Tabla pivote: promedio de gp_ratio (presión salarial) por unidad y año
pivote = df_clean.pivot_table(
    index='unidad',
    columns='anio',
    values='gp_ratio',
    aggfunc='mean'
).round(3)
print("gp_ratio (gastos personal / ingresos) por unidad y año:")
pivote

# Tasa de riesgo (label=1) por unidad
tasa_riesgo = df_clean.groupby('unidad')['label'].mean().sort_values(ascending=False)
print("Tasa de Riesgo por Unidad (proporción de periodos en riesgo):")
print(tasa_riesgo.round(3))


# Aplicamos One-Hot Encoding a la columna 'unidad'
df_encoded = pd.get_dummies(df_clean, columns=['unidad'])
print(f"Columnas tras OHE: {df_encoded.shape[1]}")
df_encoded.head()

# Resumen final del dataset limpio
print("=== DATASET LIMPIO — RESUMEN FINAL ===")
print(f"Filas:        {df_clean.shape[0]}")
print(f"Columnas:     {df_clean.shape[1]}")
print(f"Nulos:        {df_clean.isnull().sum().sum()}")
print(f"Duplicados:   {df_clean.duplicated().sum()}")
print(f"\nDistribución del target:")
print(df_clean['label'].value_counts().rename({0:'Bajo Riesgo (0)', 1:'Alto Riesgo (1)'}))