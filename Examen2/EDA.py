import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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
print(f"Q1={Q1/1e9:.2f}  Q2={Q2/1e9:.2f}  Q3={Q3/1e9:.2f}  IQR={IQR/1e9:.2f}  (miles de millones COP)")
print(f"Outliers detectados: {len(outliers)}")
print(outliers[['anio','unidad','ingresos_totales']])

percentiles = df['ingresos_totales'].quantile([0.10,0.25,0.50,0.70,0.90])/1e9
print("Percentiles de Ingresos Totales (miles de millones COP):")
print(percentiles.round(2))

plt.figure(figsize=(12,5))
order = df.groupby('unidad')['ingresos_totales'].median().sort_values().index
sns.boxplot(data=df, x='unidad', y='ingresos_totales',
            order=order, palette='Blues', hue='unidad', legend=False)
plt.xticks(rotation=30, ha='right')
plt.title('Distribución de Ingresos Totales por Unidad')
plt.ylabel('Ingresos (COP)')
plt.tight_layout();

# %%
fig, axes = plt.subplots(1, 2, figsize=(13,4))
axes[0].hist(df['ingresos_totales']/1e9, bins=15, color='steelblue', edgecolor='white')
axes[0].axvline(Q1/1e9, color='orange', linestyle='--', label='Q1/Q3')
axes[0].axvline(Q3/1e9, color='orange', linestyle='--')
axes[0].set_title('Histograma — Ingresos Totales')
axes[0].set_xlabel('Miles de millones COP'); axes[0].legend()
axes[1].boxplot(df['ingresos_totales']/1e9, vert=False, patch_artist=True,
                boxprops=dict(facecolor='steelblue', alpha=0.7))
axes[1].set_title('Boxplot — Ingresos Totales')
axes[1].set_xlabel('Miles de millones COP')
plt.tight_layout();

ingresos_unidad = (df.groupby('unidad')['ingresos_totales'].sum()/1e9).sort_values()
ingresos_unidad.plot(kind='barh', color='steelblue', edgecolor='white', figsize=(10,5))
plt.title('Ingresos Totales Acumulados por Unidad (2016-2025)')
plt.xlabel('Miles de millones COP')
plt.tight_layout();

prop = df.groupby('unidad')['ingresos_totales'].sum()
plt.figure(figsize=(8,8))
plt.pie(prop.values, labels=prop.index, autopct='%1.1f%%', startangle=90,
        colors=sns.color_palette('Blues_d', len(prop)))
plt.title('Proporción de Ingresos Totales por Unidad (2016-2025)')
plt.tight_layout();

colores = df['label'].map({0:'steelblue', 1:'tomato'})
plt.figure(figsize=(9,5))
plt.scatter(df['gastos_personal']/1e9, df['ingresos_totales']/1e9,
            c=colores, alpha=0.8, edgecolors='white')
plt.xlabel('Gastos de Personal (miles de millones COP)')
plt.ylabel('Ingresos Totales (miles de millones COP)')
plt.title('Gastos de Personal vs. Ingresos Totales')
from matplotlib.patches import Patch
plt.legend(handles=[Patch(color='steelblue', label='Bajo Riesgo (0)'),
                    Patch(color='tomato',     label='Alto Riesgo (1)')])
plt.tight_layout();

num_df = df.select_dtypes(include=np.number).drop(columns=['anio'])
corr = num_df.corr()
plt.figure(figsize=(12,9))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0,
            annot=True, fmt='.2f', linewidths=0.5, annot_kws={'size':8})
plt.title('Matriz de Correlación — FIRE UdeA', fontsize=13)
plt.tight_layout();

variables = ['liquidez','endeudamiento','gp_ratio','hhi_fuentes','tendencia_ingresos']
group_means = df.groupby('label')[variables].mean()
group_norm  = (group_means - group_means.min()) / (group_means.max() - group_means.min())
num_vars = len(variables)
angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]
colors_sp = {0:'steelblue', 1:'tomato'}
labels_n  = {0:'Bajo Riesgo', 1:'Alto Riesgo'}
fig, ax = plt.subplots(figsize=(7,7), subplot_kw=dict(polar=True))
for label_val, row in group_norm.iterrows():
    vals = row.tolist() + row.tolist()[:1]
    ax.plot(angles, vals, color=colors_sp[label_val], lw=2, label=labels_n[label_val])
    ax.fill(angles, vals, color=colors_sp[label_val], alpha=0.2)
ax.set_xticks(angles[:-1]); ax.set_xticklabels(variables, fontsize=10)
ax.set_title('Indicadores Financieros por Nivel de Riesgo', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3,1.1))
plt.tight_layout();

plt.figure(figsize=(14,5))
for unidad, grp in df.groupby('unidad'):
    plt.plot(grp['anio'], grp['ingresos_totales']/1e9, marker='o', lw=1.8, label=unidad)
plt.title('Evolución de Ingresos Totales por Unidad (2016-2025)')
plt.xlabel('Año'); plt.ylabel('Ingresos (Miles de millones COP)')
plt.legend(fontsize=8, ncol=2)
plt.tight_layout();

try:
    from ydata_profiling import ProfileReport
    profile = ProfileReport(df, title="Reporte EDA — FIRE UdeA", explorative=True)
    profile.to_notebook_iframe()
    profile.to_file("reporte_EDA_FIRE_UdeA.html")
    print("✅ Reporte guardado en 'reporte_EDA_FIRE_UdeA.html'")
except ImportError:
    print("Instala ydata-profiling con: pip install ydata-profiling")