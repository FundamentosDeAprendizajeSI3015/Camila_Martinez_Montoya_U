import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

os.makedirs("graficas", exist_ok=True)

plt.rc('font', family='serif', size=12)
sns.set_style('whitegrid')

# Cargar el dataset limpio
df = pd.read_csv('dataset_limpio.csv')
print("DataFrame limpio:")
print(df.head(10))
print(f"\nShape: {df.shape}")

df.describe(include='all').T

print("Tipos de dato:")
print(df.dtypes)
print("\nNulos:", df.isnull().sum().sum())

cols_analizar = ['ingresos_totales','gastos_personal','liquidez','endeudamiento','gp_ratio']
print("--- Media ---")
print((df[cols_analizar].mean()/1e9).round(3))
print("\n--- Mediana ---")
print((df[cols_analizar].median()/1e9).round(3))
print("\n--- Desviación Estándar ---")
print((df[cols_analizar].std()/1e9).round(3))

Q1 = df['ingresos_totales'].quantile(0.25)
Q2 = df['ingresos_totales'].quantile(0.50)
Q3 = df['ingresos_totales'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['ingresos_totales'] < lower_bound) | (df['ingresos_totales'] > upper_bound)]
print(f"Q1={Q1/1e9:.2f}  Q2={Q2/1e9:.2f}  Q3={Q3/1e9:.2f}  IQR={IQR/1e9:.2f}")
print(f"Outliers detectados: {len(outliers)}")

percentiles = df['ingresos_totales'].quantile([0.10,0.25,0.50,0.70,0.90])/1e9
print("Percentiles:")
print(percentiles.round(2))

# =========================
# TUS GRÁFICAS (AHORA GUARDADAS)
# =========================

plt.figure(figsize=(12,5))
order = df.groupby('unidad')['ingresos_totales'].median().sort_values().index
sns.boxplot(data=df, x='unidad', y='ingresos_totales',
            order=order, palette='Blues', hue='unidad', legend=False)
plt.xticks(rotation=30, ha='right')
plt.title('Distribución de Ingresos Totales por Unidad')
plt.tight_layout()
plt.savefig("graficas/boxplot_unidad.png", dpi=300)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(13,4))
axes[0].hist(df['ingresos_totales']/1e9, bins=15)
axes[0].axvline(Q1/1e9, linestyle='--')
axes[0].axvline(Q3/1e9, linestyle='--')
axes[1].boxplot(df['ingresos_totales']/1e9)
plt.tight_layout()
plt.savefig("graficas/distribucion.png", dpi=300)
plt.close()

ingresos_unidad = (df.groupby('unidad')['ingresos_totales'].sum()/1e9).sort_values()
ingresos_unidad.plot(kind='barh', figsize=(10,5))
plt.tight_layout()
plt.savefig("graficas/barras.png", dpi=300)
plt.close()

prop = df.groupby('unidad')['ingresos_totales'].sum()
plt.figure(figsize=(8,8))
plt.pie(prop.values, labels=prop.index, autopct='%1.1f%%')
plt.tight_layout()
plt.savefig("graficas/pie.png", dpi=300)
plt.close()

colores = df['label'].map({0:'steelblue', 1:'tomato'})
plt.figure(figsize=(9,5))
plt.scatter(df['gastos_personal']/1e9, df['ingresos_totales']/1e9,
            c=colores, alpha=0.8)
plt.title("Scatter básico")
plt.tight_layout()
plt.savefig("graficas/scatter_basico.png", dpi=300)
plt.close()

num_df = df.select_dtypes(include=np.number).drop(columns=['anio'])
corr = num_df.corr()
plt.figure(figsize=(12,9))
sns.heatmap(corr, annot=True)
plt.tight_layout()
plt.savefig("graficas/correlacion.png", dpi=300)
plt.close()

# =========================
# 🔥 NUEVAS GRÁFICAS CLAVE (PRO)
# =========================

# 1. Scatter mejorado
plt.figure(figsize=(9,5))
sns.scatterplot(data=df,
                x=df['gastos_personal']/1e9,
                y=df['ingresos_totales']/1e9,
                hue='label',
                palette={0:'steelblue',1:'tomato'})
plt.title("Relación Gastos vs Ingresos (con Riesgo)")
plt.tight_layout()
plt.savefig("graficas/scatter_pro.png", dpi=300)
plt.close()

# 2. Correlación PRO
plt.figure(figsize=(10,8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", center=0)
plt.title("Correlación optimizada")
plt.tight_layout()
plt.savefig("graficas/correlacion_pro.png", dpi=300)
plt.close()

# 3. Radar (ya lo tenías, ahora guardado)
variables = ['liquidez','endeudamiento','gp_ratio','hhi_fuentes','tendencia_ingresos']
group_means = df.groupby('label')[variables].mean()
group_norm  = (group_means - group_means.min()) / (group_means.max() - group_means.min())

angles = np.linspace(0, 2*np.pi, len(variables), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7,7), subplot_kw=dict(polar=True))
for i, row in group_norm.iterrows():
    vals = row.tolist() + row.tolist()[:1]
    ax.plot(angles, vals, label=f"Riesgo {i}")
    ax.fill(angles, vals, alpha=0.2)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(variables)
plt.title("Radar Riesgo")
plt.legend()

plt.savefig("graficas/radar.png", dpi=300)
plt.close()

# 4. Serie de tiempo (guardada)
plt.figure(figsize=(14,5))
for unidad, grp in df.groupby('unidad'):
    plt.plot(grp['anio'], grp['ingresos_totales']/1e9, label=unidad)
plt.title("Serie de tiempo")
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig("graficas/serie_tiempo.png", dpi=300)
plt.close()

# =========================
# REPORTE
# =========================
try:
    from ydata_profiling import ProfileReport
    profile = ProfileReport(df)
    profile.to_file("reporte.html")
except:
    pass