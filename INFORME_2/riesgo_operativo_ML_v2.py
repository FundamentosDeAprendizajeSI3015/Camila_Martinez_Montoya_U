# ==============================================================================
# PROYECTO: Riesgo Operativo en Equipos de Trabajo
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Preprocesamiento y pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import (train_test_split, RandomizedSearchCV,
                                     StratifiedKFold, cross_val_score)
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from scipy.stats import reciprocal

# Reducción de dimensionalidad
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap_lib

# Clustering No Supervisado
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (silhouette_score, adjusted_rand_score,
                             mean_absolute_error, mean_squared_error,
                             f1_score, accuracy_score,
                             confusion_matrix, ConfusionMatrixDisplay,
                             classification_report, roc_auc_score,
                             roc_curve)
import skfuzzy as fuzz
import hdbscan

# Modelos Supervisados
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, Ridge, Lasso

# Estilo del curso
plt.rc('font', family='serif', size=12)
sns.set(style='whitegrid', context='talk')
random_state = 42
np.random.seed(random_state)

# ==============================================================================
# 0. CONFIGURACIÓN
# ==============================================================================
CSV_PATH  = 'dataset_riesgo_operativo_ampliado.csv'
FEATURES  = ['t_prom', 'pct_fuera', 'pct_cumplimiento',
             'interaccion', 'des_carga', 'correcciones']
TARGET    = 'riesgo_final'
SAVE_FIG  = True   # Pon False si no quieres guardar PNGs

def savefig(name):
    if SAVE_FIG:
        plt.savefig(f'{name}.png', dpi=130, bbox_inches='tight')
    plt.show()

# ==============================================================================
# 1. CARGA Y EDA
# ==============================================================================
print("=" * 70)
print("1. CARGA Y ANÁLISIS EXPLORATORIO")
print("=" * 70)

df = pd.read_csv(CSV_PATH)
print(f"\n[INFO] Dataset: {df.shape[0]} filas × {df.shape[1]} columnas")
print(df.info())
print("\nEstadísticas descriptivas:")
print(df.describe().round(4))
print(f"\nValores nulos: {df.isnull().sum().sum()}")

# --- Crear variable de clase (terciles) ---
q33 = df[TARGET].quantile(0.33)
q66 = df[TARGET].quantile(0.66)
df['riesgo_clase'] = pd.cut(
    df[TARGET], bins=[-np.inf, q33, q66, np.inf],
    labels=['Bajo', 'Medio', 'Alto']
).astype(str)
label_map = {'Bajo': 0, 'Medio': 1, 'Alto': 2}
inv_map   = {0: 'Bajo', 1: 'Medio', 2: 'Alto'}

print(f"\nUmbrales: Bajo ≤ {q33:.3f} | Medio ≤ {q66:.3f} | Alto > {q66:.3f}")
print("\nDistribución de clases:")
print(df['riesgo_clase'].value_counts())

# --- Outliers con IQR ---
print("\n--- Outliers (regla 1.5·IQR) ---")
for col in FEATURES + [TARGET]:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    n_out = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
    print(f"  {col:22s}: {n_out:3d} outliers")

# --- Correlaciones ---
print("\n--- Correlaciones con riesgo_final ---")
corr = df[FEATURES + [TARGET]].corr()[TARGET].sort_values(ascending=False)
print(corr.round(3))

# --- Gráfica 1: Histogramas ---
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
for i, col in enumerate(FEATURES):
    axes[i].hist(df[col], bins=30, color='steelblue', edgecolor='black', alpha=0.85)
    axes[i].set_title(col, fontsize=13)
    axes[i].set_xlabel('Valor')
    axes[i].set_ylabel('Frecuencia')
plt.suptitle('Distribución de variables — Riesgo Operativo', fontsize=15, y=1.01)
plt.tight_layout()
savefig('fig01_histogramas')

# --- Gráfica 2: Heatmap correlación ---
fig, ax = plt.subplots(figsize=(9, 7))
mask = np.zeros_like(df[FEATURES + [TARGET]].corr())
np.fill_diagonal(mask, True)
sns.heatmap(df[FEATURES + [TARGET]].corr(), annot=True, fmt='.2f',
            cmap='coolwarm', ax=ax, linewidths=0.5, mask=mask.astype(bool))
mask2 = np.eye(len(FEATURES) + 1, dtype=bool)
sns.heatmap(df[FEATURES + [TARGET]].corr(), annot=True, fmt='.2f',
            cmap='coolwarm', ax=ax, linewidths=0.5)
ax.set_title('Matriz de Correlación')
plt.tight_layout()
savefig('fig02_correlacion')

# --- Gráfica 3: Boxplot por clase ---
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
orden = ['Bajo', 'Medio', 'Alto']
colores = {'Bajo': '#2196F3', 'Medio': '#FF9800', 'Alto': '#F44336'}
for i, col in enumerate(FEATURES):
    data_by_class = [df[df['riesgo_clase'] == cls][col].values for cls in orden]
    bp = axes[i].boxplot(data_by_class, labels=orden, patch_artist=True)
    for patch, cls in zip(bp['boxes'], orden):
        patch.set_facecolor(colores[cls])
    axes[i].set_title(col, fontsize=13)
    axes[i].set_xlabel('Clase de riesgo')
plt.suptitle('Distribución de variables por clase de riesgo', fontsize=15, y=1.01)
plt.tight_layout()
savefig('fig03_boxplot_clase')

# --- Gráfica 4: riesgo_final por equipo ---
fig, ax = plt.subplots(figsize=(14, 6))
df.groupby('equipo')['riesgo_final'].mean().plot(kind='bar', ax=ax,
    color='steelblue', edgecolor='black')
ax.axhline(df['riesgo_final'].mean(), color='red', linestyle='--', label='Media global')
ax.set_title('Riesgo promedio por equipo')
ax.set_xlabel('Equipo')
ax.set_ylabel('Riesgo promedio')
ax.legend()
plt.tight_layout()
savefig('fig04_riesgo_por_equipo')


# ==============================================================================
# 2. PREPROCESAMIENTO PARA CLUSTERING
# ==============================================================================
print("\n" + "=" * 70)
print("2. PREPROCESAMIENTO")
print("=" * 70)

X_raw    = df[FEATURES].values
y_orig   = df['riesgo_clase'].map(label_map).values

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# ---- Reducción de dimensionalidad: PCA, UMAP, t-SNE ----
print("\n[INFO] Calculando PCA, UMAP y t-SNE...")

# PCA
pca_full = PCA().fit(X_scaled)
var_exp  = np.cumsum(pca_full.explained_variance_ratio_)
pca2     = PCA(n_components=2, random_state=random_state)
X_pca    = pca2.fit_transform(X_scaled)
print(f"  PCA 2D varianza explicada: {pca2.explained_variance_ratio_.sum():.2%}")
print(f"  PCA varianza acumulada por componente: {var_exp.round(3)}")

# UMAP
reducer_umap = umap_lib.UMAP(n_components=2, n_neighbors=15,
                              min_dist=0.1, random_state=random_state)
X_umap = reducer_umap.fit_transform(X_scaled)
print("  UMAP calculado ✓")

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=random_state)
X_tsne = tsne.fit_transform(X_scaled)
print("  t-SNE calculado ✓")

# --- Gráfica 5: Varianza explicada PCA ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(range(1, len(var_exp)+1), pca_full.explained_variance_ratio_,
       alpha=0.7, color='steelblue', label='Varianza por componente')
ax.step(range(1, len(var_exp)+1), var_exp, where='mid',
        color='red', label='Varianza acumulada')
ax.axhline(0.80, color='gray', linestyle='--', alpha=0.7, label='80%')
ax.set_xlabel('Componente Principal')
ax.set_ylabel('Proporción de varianza')
ax.set_title('Varianza Explicada — PCA')
ax.legend()
plt.tight_layout()
savefig('fig05_pca_varianza')

def plot_dim_reduction(X_2d, labels, title, filename, label_names=None, cmap='tab10'):
    """Visualiza proyección 2D coloreada por cluster o clase."""
    unique_labels = sorted(set(labels))
    colors = plt.cm.get_cmap(cmap, len(unique_labels))
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, lbl in enumerate(unique_labels):
        mask = np.array(labels) == lbl
        name = label_names[lbl] if label_names else str(lbl)
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], s=20, alpha=0.7,
                   color=colors(i), label=name)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    savefig(filename)

# --- Gráfica 6: Clases reales en PCA, UMAP, t-SNE ---
label_names_clase = {0: 'Bajo', 1: 'Medio', 2: 'Alto'}
plot_dim_reduction(X_pca,  y_orig, 'Clases de riesgo — PCA 2D',   'fig06a_pca_clases',  label_names_clase)
plot_dim_reduction(X_umap, y_orig, 'Clases de riesgo — UMAP 2D',  'fig06b_umap_clases', label_names_clase)
plot_dim_reduction(X_tsne, y_orig, 'Clases de riesgo — t-SNE 2D', 'fig06c_tsne_clases', label_names_clase)


# ==============================================================================
# 3. CLUSTERING NO SUPERVISADO
# ==============================================================================
print("\n" + "=" * 70)
print("3. CLUSTERING NO SUPERVISADO")
print("=" * 70)

silhouette_resumen = {}

# ---- 3.1 K-MEANS — Método del Codo + Silhouette ---------------------------
print("\n--- 3.1 K-Means ---")

inert_list, sil_list = [], []
k_range = range(2, 11)
for k in k_range:
    km = KMeans(n_clusters=k, n_init=15, random_state=random_state)
    lbl = km.fit_predict(X_scaled)
    inert_list.append(km.inertia_)
    sil_list.append(silhouette_score(X_scaled, lbl))

# --- Gráfica 7: Codo + Silhouette ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(list(k_range), inert_list, 'bo-', linewidth=2)
axes[0].set_xlabel('K')
axes[0].set_ylabel('Inercia (WCSS)')
axes[0].set_title('Método del Codo — K-Means')
axes[1].plot(list(k_range), sil_list, 'rs-', linewidth=2)
axes[1].set_xlabel('K')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score — K-Means')
plt.suptitle('Selección de K óptimo')
plt.tight_layout()
savefig('fig07_kmeans_codo')

K_opt = list(k_range)[np.argmax(sil_list)]
print(f"[INFO] K óptimo (mayor Silhouette): {K_opt}")

# Ajuste final con K óptimo
km_final = KMeans(n_clusters=K_opt, n_init=20, random_state=random_state)
labels_km = km_final.fit_predict(X_scaled)
sil_km    = silhouette_score(X_scaled, labels_km)
silhouette_resumen['K-Means'] = sil_km
print(f"K-Means (K={K_opt}) — Inercia: {km_final.inertia_:.1f} | Silhouette: {sil_km:.4f}")
print(pd.Series(labels_km).value_counts().sort_index())

# --- Gráfica 8: K-Means en PCA, UMAP, t-SNE ---
plot_dim_reduction(X_pca,  labels_km, f'K-Means (K={K_opt}) — PCA',   'fig08a_kmeans_pca')
plot_dim_reduction(X_umap, labels_km, f'K-Means (K={K_opt}) — UMAP',  'fig08b_kmeans_umap')
plot_dim_reduction(X_tsne, labels_km, f'K-Means (K={K_opt}) — t-SNE', 'fig08c_kmeans_tsne')
df['cluster_kmeans'] = labels_km


# ---- 3.2 FUZZY C-MEANS -------------------------------------------------------
print("\n--- 3.2 Fuzzy C-Means ---")

X_fcm = X_scaled.T
best_sil_fcm, best_c, best_lbl_fcm = -1, 2, None

for c in range(2, 8):
    cntr, u, *_ = fuzz.cluster.cmeans(
        X_fcm, c=c, m=2.0, error=0.005, maxiter=1000, seed=random_state)
    lbl_fcm = np.argmax(u, axis=0)
    sil = silhouette_score(X_scaled, lbl_fcm)
    if sil > best_sil_fcm:
        best_sil_fcm, best_c, best_lbl_fcm = sil, c, lbl_fcm

silhouette_resumen['Fuzzy C-Means'] = best_sil_fcm
print(f"Fuzzy C-Means (C={best_c}) — Silhouette: {best_sil_fcm:.4f}")
print(pd.Series(best_lbl_fcm).value_counts().sort_index())

plot_dim_reduction(X_pca,  best_lbl_fcm, f'Fuzzy C-Means (C={best_c}) — PCA',   'fig09a_fcm_pca')
plot_dim_reduction(X_umap, best_lbl_fcm, f'Fuzzy C-Means (C={best_c}) — UMAP',  'fig09b_fcm_umap')
plot_dim_reduction(X_tsne, best_lbl_fcm, f'Fuzzy C-Means (C={best_c}) — t-SNE', 'fig09c_fcm_tsne')
df['cluster_fcm'] = best_lbl_fcm


# ---- 3.3 SUBTRACTIVE CLUSTERING ----------------------------------------------
print("\n--- 3.3 Subtractive Clustering ---")

def subtractive_clustering(X, ra=1.5, rb_factor=1.5, accept=0.5, reject=0.15):
    """Implementación Subtractive Clustering (Chiu, 1994)."""
    rb = ra * rb_factor
    X  = np.array(X, dtype=float)
    n  = len(X)
    potential = np.array([
        np.sum(np.exp(-np.sum((X - X[i])**2, axis=1) / (ra/2)**2))
        for i in range(n)
    ])
    centers, labels, cluster_id = [], -np.ones(n, dtype=int), 0
    p = potential.copy()
    p0_max = p.max()

    while True:
        idx_max = np.argmax(p)
        p_max   = p[idx_max]
        if p_max < reject * p0_max or cluster_id > 25:
            break
        if p_max < accept * p0_max and cluster_id > 0:
            break
        centers.append(X[idx_max])
        dists = np.sum((X - X[idx_max])**2, axis=1)
        p -= p_max * np.exp(-dists / (rb/2)**2)
        cluster_id += 1

    centers = np.array(centers)
    if len(centers) == 0:
        return np.zeros(n, dtype=int), centers
    for i in range(n):
        dists = np.sum((centers - X[i])**2, axis=1)
        labels[i] = np.argmin(dists)
    return labels, centers

labels_sub, centers_sub = subtractive_clustering(X_scaled, ra=1.5)
n_cls_sub = len(np.unique(labels_sub))
sil_sub   = silhouette_score(X_scaled, labels_sub) if n_cls_sub > 1 else 0
silhouette_resumen['Subtractive'] = sil_sub
print(f"Subtractive — Clusters: {n_cls_sub} centros | Silhouette: {sil_sub:.4f}")

plot_dim_reduction(X_pca,  labels_sub, f'Subtractive ({n_cls_sub} centros) — PCA',   'fig10a_sub_pca')
plot_dim_reduction(X_umap, labels_sub, f'Subtractive ({n_cls_sub} centros) — UMAP',  'fig10b_sub_umap')
plot_dim_reduction(X_tsne, labels_sub, f'Subtractive ({n_cls_sub} centros) — t-SNE', 'fig10c_sub_tsne')
df['cluster_subtractive'] = labels_sub


# ---- 3.4 DBSCAN --------------------------------------------------------------
print("\n--- 3.4 DBSCAN ---")

# k-distance plot
k_nn = 5
nn = NearestNeighbors(n_neighbors=k_nn).fit(X_scaled)
dists, _ = nn.kneighbors(X_scaled)
k_dists  = np.sort(dists[:, k_nn-1])

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(k_dists, color='steelblue', linewidth=1.5)
ax.set_xlabel('Observaciones ordenadas')
ax.set_ylabel(f'Distancia al {k_nn}° vecino')
ax.set_title('k-distance plot — Selección de eps para DBSCAN')
plt.tight_layout()
savefig('fig11_dbscan_kdist')

# Grid search eps / min_samples — requiere ≥50% puntos no ruido
best_sil_db, best_eps, best_ms, best_lbl_db = -1, 0.5, 5, None
for eps in np.arange(0.3, 2.5, 0.1):
    for ms in [3, 5, 8, 10, 15]:
        lbls = DBSCAN(eps=round(eps, 1), min_samples=ms).fit_predict(X_scaled)
        n_u  = len(set(lbls)) - (1 if -1 in lbls else 0)
        if n_u < 2:
            continue
        mask = lbls != -1
        if mask.sum() < len(lbls) * 0.5:
            continue
        sil = silhouette_score(X_scaled[mask], lbls[mask])
        if sil > best_sil_db:
            best_sil_db, best_eps, best_ms, best_lbl_db = sil, round(eps,1), ms, lbls

n_noise_db = np.sum(best_lbl_db == -1)
n_cls_db   = len(set(best_lbl_db)) - (1 if -1 in best_lbl_db else 0)
silhouette_resumen['DBSCAN'] = best_sil_db
print(f"DBSCAN (eps={best_eps}, min_samples={best_ms}) — Clusters: {n_cls_db} | Ruido: {n_noise_db} | Silhouette: {best_sil_db:.4f}")

plot_dim_reduction(X_pca,  best_lbl_db, f'DBSCAN (eps={best_eps}, min={best_ms}) — PCA',   'fig12a_dbscan_pca')
plot_dim_reduction(X_umap, best_lbl_db, f'DBSCAN (eps={best_eps}, min={best_ms}) — UMAP',  'fig12b_dbscan_umap')
plot_dim_reduction(X_tsne, best_lbl_db, f'DBSCAN (eps={best_eps}, min={best_ms}) — t-SNE', 'fig12c_dbscan_tsne')
df['cluster_dbscan'] = best_lbl_db


# ---- 3.5 FAMILIA AGGLOMERATIVE (Ward, Average, Complete) ---------------------
print("\n--- 3.5 Familia Agglomerative Clustering ---")

aggl_results = {}
for linkage in ['ward', 'average', 'complete']:
    best_sil_ag, best_k_ag, best_lbl_ag = -1, 2, None
    for k in range(2, 9):
        model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
        lbls  = model.fit_predict(X_scaled)
        sil   = silhouette_score(X_scaled, lbls)
        if sil > best_sil_ag:
            best_sil_ag, best_k_ag, best_lbl_ag = sil, k, lbls
    aggl_results[linkage] = {'sil': best_sil_ag, 'k': best_k_ag, 'labels': best_lbl_ag}
    silhouette_resumen[f'Agglom. {linkage.capitalize()}'] = best_sil_ag
    print(f"  {linkage.capitalize():10s}: K={best_k_ag} | Silhouette={best_sil_ag:.4f}")

best_aggl_name = max(aggl_results, key=lambda x: aggl_results[x]['sil'])
best_lbl_aggl  = aggl_results[best_aggl_name]['labels']
best_k_aggl    = aggl_results[best_aggl_name]['k']

plot_dim_reduction(X_pca,  best_lbl_aggl,
                   f'Agglom. {best_aggl_name.capitalize()} (K={best_k_aggl}) — PCA',
                   'fig13a_aggl_pca')
plot_dim_reduction(X_umap, best_lbl_aggl,
                   f'Agglom. {best_aggl_name.capitalize()} (K={best_k_aggl}) — UMAP',
                   'fig13b_aggl_umap')
plot_dim_reduction(X_tsne, best_lbl_aggl,
                   f'Agglom. {best_aggl_name.capitalize()} (K={best_k_aggl}) — t-SNE',
                   'fig13c_aggl_tsne')
df['cluster_aggl'] = best_lbl_aggl


# ---- 3.6 HDBSCAN -------------------------------------------------------------
print("\n--- 3.6 HDBSCAN ---")
best_sil_hdb, best_mcs, best_lbl_hdb = -1, 10, None
for mcs in [3, 5, 8, 10, 15, 20, 25, 30]:
    lbls = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=1).fit_predict(X_scaled)
    n_u  = len(set(lbls)) - (1 if -1 in lbls else 0)
    if n_u < 2:
        continue
    mask = lbls != -1
    if mask.sum() < len(lbls) * 0.3:
        continue
    sil = silhouette_score(X_scaled[mask], lbls[mask])
    if sil > best_sil_hdb:
        best_sil_hdb, best_mcs, best_lbl_hdb = sil, mcs, lbls

# Fallback: si HDBSCAN no converge, usar K-Means k=2
if best_lbl_hdb is None:
    best_lbl_hdb = KMeans(n_clusters=2, n_init=10, random_state=random_state).fit_predict(X_scaled)
    best_sil_hdb = silhouette_score(X_scaled, best_lbl_hdb)
    best_mcs = 'N/A (fallback KMeans k=2)'
    print(f"  [WARN] HDBSCAN sin resultado válido, usando KMeans k=2 como fallback")

n_noise_hdb = int(np.sum(best_lbl_hdb == -1))
n_cls_hdb   = len(set(best_lbl_hdb)) - (1 if -1 in best_lbl_hdb else 0)
silhouette_resumen['HDBSCAN'] = best_sil_hdb
print(f"HDBSCAN (min_cs={best_mcs}) — Clusters: {n_cls_hdb} | Ruido: {n_noise_hdb} | Silhouette: {best_sil_hdb:.4f}")

plot_dim_reduction(X_pca,  best_lbl_hdb, f'HDBSCAN (min_cs={best_mcs}) — PCA',   'fig14a_hdbscan_pca')
plot_dim_reduction(X_umap, best_lbl_hdb, f'HDBSCAN (min_cs={best_mcs}) — UMAP',  'fig14b_hdbscan_umap')
plot_dim_reduction(X_tsne, best_lbl_hdb, f'HDBSCAN (min_cs={best_mcs}) — t-SNE', 'fig14c_hdbscan_tsne')
df['cluster_hdbscan'] = best_lbl_hdb


# ---- 3.7 Resumen comparativo clustering --------------------------------------
print("\n--- 3.7 Resumen Silhouette por método ---")
df_sil = pd.DataFrame(list(silhouette_resumen.items()),
                      columns=['Método', 'Silhouette']).sort_values('Silhouette', ascending=False)
print(df_sil.to_string(index=False))

fig, ax = plt.subplots(figsize=(11, 5))
colores_bar = ['#2196F3' if s == df_sil['Silhouette'].max() else '#90CAF9'
               for s in df_sil['Silhouette']]
ax.barh(df_sil['Método'], df_sil['Silhouette'], color=colores_bar, edgecolor='black')
ax.set_xlabel('Silhouette Score')
ax.set_title('Comparación de métodos de clustering — Silhouette Score')
ax.axvline(0, color='black', linewidth=0.8)
for i, (met, sil) in enumerate(zip(df_sil['Método'], df_sil['Silhouette'])):
    ax.text(sil + 0.002, i, f'{sil:.3f}', va='center', fontsize=10)
plt.tight_layout()
savefig('fig15_silhouette_resumen')


# ==============================================================================
# 4. RE-EVALUACIÓN DE ETIQUETAS
# ==============================================================================
print("\n" + "=" * 70)
print("4. RE-EVALUACIÓN DE ETIQUETAS (máx. 30% re-etiquetado)")
print("=" * 70)
print("""
Estrategia:
  1. KNN (k=11) entrena sobre el espacio escalado para estimar la
     etiqueta de "consenso vecinal" de cada punto.
  2. Se identifica qué etiquetas discrepan con ese consenso.
  3. De los discrepantes, se re-etiquetan solo los de mayor incertidumbre
     (menor probabilidad de su clase) hasta un máximo del 30%.
  4. Se reporta ARI y alineación como métricas de calidad del relabeling.
""")

MAX_RELABEL = int(len(y_orig) * 0.30)

# KNN consenso (k=11 votos mayoritarios ponderados)
knn_rel = KNeighborsClassifier(n_neighbors=11, weights='distance')
knn_rel.fit(X_scaled, y_orig)
y_consenso = knn_rel.predict(X_scaled)
proba      = knn_rel.predict_proba(X_scaled)
confidence = proba.max(axis=1)
uncertainty= 1 - confidence

disc_mask  = y_orig != y_consenso
n_disc     = disc_mask.sum()
pct_disc   = disc_mask.mean()

# ARI entre consenso KNN y etiquetas originales
ari_val    = adjusted_rand_score(y_orig, y_consenso)
alignment  = (y_orig == y_consenso).mean()

print(f"Alineación KNN vs etiquetas originales : {alignment:.1%}")
print(f"Adjusted Rand Index (ARI)              : {ari_val:.4f}")
print(f"Muestras discrepantes                  : {n_disc} ({pct_disc:.1%})")
print(f"Máximo re-etiquetable (30%)            : {MAX_RELABEL}")

# Priorizar por mayor incertidumbre
disc_indices = np.where(disc_mask)[0]
unc_disc     = uncertainty[disc_indices]
sorted_disc  = disc_indices[np.argsort(-unc_disc)]
to_relabel   = sorted_disc[:MAX_RELABEL]

y_clean = y_orig.copy()
y_clean[to_relabel] = y_consenso[to_relabel]

n_relabeled = len(to_relabel)
print(f"\nMuestras re-etiquetadas                : {n_relabeled} ({n_relabeled/len(y_orig):.1%})")
print(f"\nBalance ORIGINAL: {pd.Series(y_orig).map(inv_map).value_counts().to_dict()}")
print(f"Balance LIMPIO:   {pd.Series(y_clean).map(inv_map).value_counts().to_dict()}")

# Marcar los re-etiquetados en el df
df['riesgo_clase_orig']  = pd.Series(y_orig).map(inv_map).values
df['riesgo_clase_clean'] = pd.Series(y_clean).map(inv_map).values
relabeled_mask = np.zeros(len(df), dtype=bool)
relabeled_mask[to_relabel] = True

# --- Gráfica 16: Original vs Re-etiquetado en PCA (círculos dorados) ---
for proj, X2d, fname in [('PCA', X_pca, 'fig16a'), ('UMAP', X_umap, 'fig16b'), ('t-SNE', X_tsne, 'fig16c')]:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    cmap_cls = {0: '#2196F3', 1: '#FF9800', 2: '#F44336'}
    for ax_idx, (y_use, title_s) in enumerate([(y_orig, f'Original ({proj})'),
                                                (y_clean, f'Re-etiquetado ({proj})')]):
        ax = axes[ax_idx]
        for cls in [0, 1, 2]:
            mask = y_use == cls
            ax.scatter(X2d[mask, 0], X2d[mask, 1], c=cmap_cls[cls],
                       s=15, alpha=0.6, label=inv_map[cls])
        if ax_idx == 1:
            ax.scatter(X2d[relabeled_mask, 0], X2d[relabeled_mask, 1],
                       s=60, facecolors='none', edgecolors='gold',
                       linewidths=1.2, label='Re-etiquetado', zorder=5)
        ax.set_title(title_s, fontsize=12)
        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')
        ax.legend(fontsize=9, bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.suptitle(f'Etiquetas Originales vs Re-etiquetadas — {proj}', fontsize=13)
    plt.tight_layout()
    savefig(f'{fname}_relabeling_{proj.lower().replace("-","_")}')


# ==============================================================================
# 5. MODELOS SUPERVISADOS
# ==============================================================================
print("\n" + "=" * 70)
print("5. MODELOS SUPERVISADOS")
print("=" * 70)

# Split estratificado 80/20
X_df = df[FEATURES]

(X_tr_o, X_te_o,
 y_cls_tr_o, y_cls_te_o,
 y_reg_tr_o, y_reg_te_o) = train_test_split(
    X_df, y_orig, df[TARGET],
    test_size=0.2, random_state=random_state, stratify=y_orig)

(X_tr_c, X_te_c,
 y_cls_tr_c, y_cls_te_c,
 y_reg_tr_c, y_reg_te_c) = train_test_split(
    X_df, y_clean, df[TARGET],
    test_size=0.2, random_state=random_state, stratify=y_clean)

print(f"Train: {len(X_tr_o)} | Test: {len(X_te_o)}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

def eval_clf(model, X_tr, X_te, y_tr, y_te, name):
    """Evalúa modelo de clasificación y devuelve métricas."""
    y_pred = model.predict(X_te)
    acc  = accuracy_score(y_te, y_pred)
    f1   = f1_score(y_te, y_pred, average='weighted', zero_division=0)
    # AUC multiclase (one-vs-rest)
    try:
        proba = model.predict_proba(X_te)
        auc  = roc_auc_score(y_te, proba, multi_class='ovr', average='weighted')
    except Exception:
        auc = float('nan')
    print(f"\n{name}")
    print(f"  Accuracy={acc:.4f} | F1={f1:.4f} | AUC={auc:.4f}")
    print(classification_report(y_te, y_pred,
                                target_names=['Bajo','Medio','Alto'],
                                zero_division=0))
    return {'Accuracy': round(acc,4), 'F1': round(f1,4), 'AUC': round(auc,4)}

resultados = []

# ---- 5.1 ÁRBOL DE DECISIÓN ---------------------------------------------------
print("\n--- 5.1 Árbol de Decisión (Clasificación) ---")

param_dt = {
    'clf__max_depth':        [3, 4, 5, 6, 8, 10, None],
    'clf__min_samples_leaf': [5, 10, 20, 30, 50],
    'clf__criterion':        ['gini', 'entropy'],
    'clf__min_samples_split':[2, 5, 10]
}

def pipe_dt():
    return Pipeline([('scaler', StandardScaler()),
                     ('clf', DecisionTreeClassifier(random_state=random_state))])

for ds_name, X_tr, X_te, y_tr, y_te in [
    ('Original', X_tr_o, X_te_o, y_cls_tr_o, y_cls_te_o),
    ('Limpio',   X_tr_c, X_te_c, y_cls_tr_c, y_cls_te_c)
]:
    search = RandomizedSearchCV(pipe_dt(), param_dt, n_iter=60, cv=cv,
                                scoring='f1_weighted', random_state=random_state, n_jobs=-1)
    search.fit(X_tr, y_tr)
    print(f"  [{ds_name}] Mejores params: {search.best_params_}")
    met = eval_clf(search, X_tr, X_te, y_tr, y_te, f"Árbol [{ds_name}]")
    met.update({'Modelo': 'Árbol de Decisión', 'Dataset': ds_name})
    resultados.append(met)

    # Matriz de confusión
    y_pred = search.predict(X_te)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_te, y_pred),
                           display_labels=['Bajo','Medio','Alto']).plot(ax=ax, colorbar=False)
    ax.set_title(f'Matriz de Confusión — Árbol [{ds_name}]')
    plt.tight_layout()
    savefig(f'fig17_cm_arbol_{ds_name.lower()}')

    # Importancia de variables (solo dataset limpio)
    if ds_name == 'Limpio':
        feat_imp = pd.Series(
            search.best_estimator_['clf'].feature_importances_,
            index=FEATURES).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        feat_imp.plot(kind='barh', ax=ax, color='steelblue', edgecolor='black')
        ax.set_title('Importancia de variables — Árbol de Decisión [Limpio]')
        ax.set_xlabel('Importancia (Gini)')
        plt.tight_layout()
        savefig('fig18_arbol_importancia')

    globals()[f'dt_{ds_name.lower()}'] = search


# ---- 5.2 REGRESIÓN LOGÍSTICA -------------------------------------------------
print("\n--- 5.2 Regresión Logística (Clasificación) ---")

param_lr = {
    'poly__degree': [1, 2, 3],
    'clf__C':       reciprocal(1e-3, 1e3),
    'clf__solver':  ['lbfgs', 'saga']
}

def pipe_lr():
    return Pipeline([
        ('scaler',  StandardScaler()),
        ('poly',    PolynomialFeatures(include_bias=False)),
        ('scaler2', StandardScaler()),
        ('clf',     LogisticRegression(max_iter=3000, random_state=random_state))
    ])

for ds_name, X_tr, X_te, y_tr, y_te in [
    ('Original', X_tr_o, X_te_o, y_cls_tr_o, y_cls_te_o),
    ('Limpio',   X_tr_c, X_te_c, y_cls_tr_c, y_cls_te_c)
]:
    search = RandomizedSearchCV(pipe_lr(), param_lr, n_iter=60, cv=cv,
                                scoring='f1_weighted', random_state=random_state, n_jobs=-1)
    search.fit(X_tr, y_tr)
    print(f"  [{ds_name}] Mejores params: {search.best_params_}")
    met = eval_clf(search, X_tr, X_te, y_tr, y_te, f"Reg.Logística [{ds_name}]")
    met.update({'Modelo': 'Reg. Logística', 'Dataset': ds_name})
    resultados.append(met)

    y_pred = search.predict(X_te)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_te, y_pred),
                           display_labels=['Bajo','Medio','Alto']).plot(ax=ax, colorbar=False)
    ax.set_title(f'Matriz de Confusión — Reg. Logística [{ds_name}]')
    plt.tight_layout()
    savefig(f'fig19_cm_lr_{ds_name.lower()}')

    globals()[f'lr_{ds_name.lower()}'] = search


# ---- 5.3 REGRESIÓN LINEAL (Ridge + Lasso) ------------------------------------
print("\n--- 5.3 Regresión Lineal — Ridge / Lasso (predicción continua) ---")

param_reg = {
    'poly__degree':     [1, 2, 3],
    'regressor__alpha': reciprocal(1e-4, 1e2)
}

def pipe_ridge():
    return Pipeline([('poly', PolynomialFeatures(include_bias=False)),
                     ('scaler', StandardScaler()),
                     ('regressor', Ridge())])

def pipe_lasso():
    return Pipeline([('poly', PolynomialFeatures(include_bias=False)),
                     ('scaler', StandardScaler()),
                     ('regressor', Lasso(max_iter=8000))])

reg_results = []
for ds_name, X_tr, X_te, y_tr, y_te in [
    ('Original', X_tr_o, X_te_o, y_reg_tr_o, y_reg_te_o),
    ('Limpio',   X_tr_c, X_te_c, y_reg_tr_c, y_reg_te_c)
]:
    for model_name, pipe_fn in [('Ridge', pipe_ridge), ('Lasso', pipe_lasso)]:
        search = RandomizedSearchCV(pipe_fn(), param_reg, n_iter=80, cv=5,
                                    random_state=random_state, n_jobs=-1)
        search.fit(X_tr, y_tr)
        r2   = search.score(X_te, y_te)
        mae  = mean_absolute_error(y_te, search.predict(X_te))
        rmse = np.sqrt(mean_squared_error(y_te, search.predict(X_te)))
        print(f"  {model_name} [{ds_name}] — R²={r2:.4f} | MAE={mae:.4f} | RMSE={rmse:.4f} | params={search.best_params_}")
        reg_results.append({'Modelo': model_name, 'Dataset': ds_name,
                            'R²': round(r2,4), 'MAE': round(mae,4), 'RMSE': round(rmse,4)})
        globals()[f'{model_name.lower()}_{ds_name.lower()}'] = search

# --- Gráfica 20: Real vs Predicho Ridge [Limpio] ---
y_pred_ridge = ridge_limpio.predict(X_te_c)
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_reg_te_c, y_pred_ridge, alpha=0.5, color='steelblue', edgecolors='navy', s=25)
lims = [min(y_reg_te_c.min(), y_pred_ridge.min())-0.01,
        max(y_reg_te_c.max(), y_pred_ridge.max())+0.01]
ax.plot(lims, lims, 'r--', label='Predicción perfecta')
ax.set_xlabel('Riesgo real')
ax.set_ylabel('Riesgo predicho')
ax.set_title('Ridge [Limpio] — Real vs Predicho')
ax.legend()
plt.tight_layout()
savefig('fig20_ridge_real_vs_pred')


# ==============================================================================
# 6. COMPARACIÓN ORIGINAL vs LIMPIO
# ==============================================================================
print("\n" + "=" * 70)
print("6. COMPARACIÓN: ORIGINAL vs DATASET LIMPIO")
print("=" * 70)

df_res_cls = pd.DataFrame(resultados)[['Modelo','Dataset','Accuracy','F1','AUC']]
df_res_reg = pd.DataFrame(reg_results)[['Modelo','Dataset','R²','MAE','RMSE']]

print("\n--- Clasificación ---")
print(df_res_cls.pivot_table(index='Modelo', columns='Dataset',
                              values=['Accuracy','F1','AUC']).round(4))

print("\n--- Regresión ---")
print(df_res_reg.pivot_table(index='Modelo', columns='Dataset',
                               values=['R²','MAE','RMSE']).round(4))

# --- Gráfica 21: Comparación F1 ---
modelos_cls = df_res_cls['Modelo'].unique()
x = np.arange(len(modelos_cls))
w = 0.35
f1_orig  = [df_res_cls[(df_res_cls['Modelo']==m) & (df_res_cls['Dataset']=='Original')]['F1'].values[0] for m in modelos_cls]
f1_clean = [df_res_cls[(df_res_cls['Modelo']==m) & (df_res_cls['Dataset']=='Limpio')]['F1'].values[0]   for m in modelos_cls]

fig, ax = plt.subplots(figsize=(9, 6))
ax.bar(x - w/2, f1_orig,  w, label='Original', color='#EF5350', edgecolor='black')
ax.bar(x + w/2, f1_clean, w, label='Limpio',   color='#42A5F5', edgecolor='black')
ax.set_xticks(x)
ax.set_xticklabels(modelos_cls, fontsize=11)
ax.set_ylabel('F1-Score (weighted)')
ax.set_title('F1-Score: Dataset Original vs Limpio')
ax.set_ylim(0, 1.05)
ax.legend()
for i, (vo, vc) in enumerate(zip(f1_orig, f1_clean)):
    ax.text(i - w/2, vo + 0.01, f'{vo:.3f}', ha='center', fontsize=9)
    ax.text(i + w/2, vc + 0.01, f'{vc:.3f}', ha='center', fontsize=9)
plt.tight_layout()
savefig('fig21_comparacion_f1')

# --- Gráfica 22: Comparación R² regresión ---
modelos_reg = df_res_reg['Modelo'].unique()
x = np.arange(len(modelos_reg))
r2_orig  = [df_res_reg[(df_res_reg['Modelo']==m) & (df_res_reg['Dataset']=='Original')]['R²'].values[0] for m in modelos_reg]
r2_clean = [df_res_reg[(df_res_reg['Modelo']==m) & (df_res_reg['Dataset']=='Limpio')]['R²'].values[0]   for m in modelos_reg]

fig, ax = plt.subplots(figsize=(9, 6))
ax.bar(x - w/2, r2_orig,  w, label='Original', color='#EF5350', edgecolor='black')
ax.bar(x + w/2, r2_clean, w, label='Limpio',   color='#42A5F5', edgecolor='black')
ax.set_xticks(x)
ax.set_xticklabels(modelos_reg, fontsize=11)
ax.set_ylabel('R²')
ax.set_title('R²: Dataset Original vs Limpio (Regresión)')
ax.set_ylim(0, 1.05)
ax.legend()
for i, (vo, vc) in enumerate(zip(r2_orig, r2_clean)):
    ax.text(i - w/2, vo + 0.005, f'{vo:.3f}', ha='center', fontsize=9)
    ax.text(i + w/2, vc + 0.005, f'{vc:.3f}', ha='center', fontsize=9)
plt.tight_layout()
savefig('fig22_comparacion_r2')


# ==============================================================================
# 7. RESUMEN FINAL Y FIGURAS
# ==============================================================================
print("\n" + "=" * 70)
print("✅ PIPELINE COMPLETO — RESUMEN")
print("=" * 70)

print("""
CLUSTERING (Silhouette Score — mayor es mejor):""")
print(df_sil.to_string(index=False))

print("""
RE-ETIQUETADO:""")
print(f"  Alineación KNN vs original : {alignment:.1%}")
print(f"  ARI                        : {ari_val:.4f}")
print(f"  Muestras re-etiquetadas    : {n_relabeled} ({n_relabeled/len(y_orig):.1%})")

print("""
CLASIFICACIÓN:""")
print(df_res_cls.to_string(index=False))

print("""
REGRESIÓN:""")
print(df_res_reg.to_string(index=False))

print("""
FIGURAS GENERADAS (úsalas en tu presentación):
  fig01  — Histogramas de variables
  fig02  — Heatmap de correlación
  fig03  — Boxplots por clase de riesgo
  fig04  — Riesgo promedio por equipo
  fig05  — Varianza explicada PCA
  fig06  — Clases reales en PCA / UMAP / t-SNE
  fig07  — Método del codo + Silhouette K-Means
  fig08  — K-Means en PCA / UMAP / t-SNE
  fig09  — Fuzzy C-Means en PCA / UMAP / t-SNE
  fig10  — Subtractive en PCA / UMAP / t-SNE
  fig11  — k-distance plot DBSCAN
  fig12  — DBSCAN en PCA / UMAP / t-SNE
  fig13  — Agglomerative en PCA / UMAP / t-SNE
  fig14  — HDBSCAN en PCA / UMAP / t-SNE
  fig15  — Silhouette comparativo (barra)
  fig16  — Original vs Re-etiquetado en PCA / UMAP / t-SNE
  fig17  — Matrices de confusión Árbol (orig / limpio)
  fig18  — Importancia de variables Árbol
  fig19  — Matrices de confusión Reg. Logística (orig / limpio)
  fig20  — Real vs Predicho Ridge [Limpio]
  fig21  — Comparación F1 Original vs Limpio
  fig22  — Comparación R² Original vs Limpio
""")
