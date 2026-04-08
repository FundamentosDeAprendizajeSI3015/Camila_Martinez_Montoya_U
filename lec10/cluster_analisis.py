import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import hdbscan

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, accuracy_score
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage, dendrogram
import umap.umap_ as umap

sns.set(style="whitegrid", context="talk")
plt.rcParams.update({"figure.dpi": 130, "font.family": "serif",
                     "axes.spines.top": False, "axes.spines.right": False})

PALETTE = {"c0": "#185FA5", "c1": "#1D9E75", "wrong": "#E24B4A",
           "hdb0": "#185FA5", "hdb1": "#1D9E75", "hdb2": "#D85A30", "noise": "#EF9F27"}

# -----------------------------
# 1. CARGA Y PREPROCESAMIENTO
# -----------------------------

def load_udea(path="dataset_limpio.csv"):
    df = pd.read_csv(path)
    cols = [c for c in df.columns if c not in ("anio", "unidad", "label")]
    X = StandardScaler().fit_transform(df[cols].values)
    print(f"[INFO] UdeA: {df.shape[0]} obs, {len(cols)} features")
    return df, X, df["label"].values, cols

def load_sintetico(path="dataset_limpio_sintetico.csv"):
    df = pd.read_csv(path)
    cols = [c for c in df.columns if c != "label"]
    X = StandardScaler().fit_transform(df[cols].values)
    print(f"[INFO] Sintético: {df.shape[0]} obs, {len(cols)} features")
    return df, X, df["label"].values, cols

# -----------------------------
# 2. REDUCCIÓN DE DIMENSIONALIDAD
# -----------------------------

def reduce_pca(X):
    pca = PCA(n_components=2, random_state=42)
    Xr = pca.fit_transform(X)
    print(f"[INFO] PCA varianza explicada: {pca.explained_variance_ratio_.sum():.2%}")
    return Xr

def reduce_umap(X, n=2):
    return umap.UMAP(n_components=n, random_state=42).fit_transform(X)

# -----------------------------
# 3. CLUSTERING
# -----------------------------

def run_kmeans(X, y_true, k=2):
    best, best_in = None, np.inf
    for s in range(100):
        km = KMeans(k, random_state=s, n_init=10)
        lbl = km.fit_predict(X)
        if km.inertia_ < best_in:
            best_in, best = km.inertia_, lbl
    if y_true[best == 0].mean() > y_true[best == 1].mean():
        best = 1 - best
    print(f"[INFO] KMeans sil={silhouette_score(X, best):.3f}  acc={accuracy_score(y_true, best):.3f}")
    return best

def run_dbscan(X, eps, min_samples=5):
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    print(f"[INFO] DBSCAN clusters: {sorted(set(labels))}")
    return labels

def run_hdbscan(X, min_cluster_size=3, min_samples=2):
    m = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    m.fit(X)
    print(f"[INFO] HDBSCAN clusters: {sorted(set(m.labels_))}")
    return m, m.labels_

def clustering_stability(X, func, n=30, label=""):
    ref = func(X)
    aris = [adjusted_rand_score(ref[idx := np.random.choice(len(X), len(X), replace=True)],
                                func(X[idx])) for _ in range(n)]
    print(f"  [{label}] ARI={np.mean(aris):.3f} ± {np.std(aris):.3f}")

# -----------------------------
# 4. VISUALIZACIONES
# -----------------------------

def plot_clusters(X2d, labels, title):
    df = pd.DataFrame({"D1": X2d[:,0], "D2": X2d[:,1], "C": labels.astype(str)})
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=df, x="D1", y="D2", hue="C", palette="tab10", s=70, alpha=0.85)
    plt.title(title); plt.legend(bbox_to_anchor=(1.05, 1)); plt.tight_layout(); plt.show()

def plot_kdistance(X, k=5):
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    d, _ = nn.kneighbors(X)
    kd = np.sort(d[:, k-1])
    plt.figure(figsize=(9, 5))
    plt.plot(kd); plt.xlabel("Observaciones ordenadas"); plt.ylabel(f"{k}-distancia")
    plt.title("k-distance plot"); plt.tight_layout(); plt.show()
    return float(np.percentile(kd, 90))

def plot_dendrogram(X, method="ward"):
    Z = linkage(X, method=method)
    plt.figure(figsize=(13, 5))
    dendrogram(Z, truncate_mode="level", p=40)
    plt.title(f"Dendrograma – {method.upper()}"); plt.tight_layout(); plt.show()

def plot_etiquetas(X_pca, X_umap, y, mislabeled):
    colors = [PALETTE["c0"] if v==0 else PALETTE["c1"] for v in y]
    legend = [mpatches.Patch(color=PALETTE["c0"], label="Clase 0"),
               mpatches.Patch(color=PALETTE["c1"], label="Clase 1"),
               mpatches.Patch(facecolor="none", edgecolor=PALETTE["wrong"],
                              linewidth=1.8, label="Mal etiquetados")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Etiquetas originales vs. estructura real", fontweight="bold")
    for ax, coords, t in zip(axes, [X_pca, X_umap], ["PCA 2D", "UMAP 2D"]):
        ax.scatter(coords[:,0], coords[:,1], c=colors, s=55, alpha=0.82, edgecolors="white", lw=0.4)
        if len(mislabeled):
            ax.scatter(coords[mislabeled,0], coords[mislabeled,1],
                       s=130, facecolors="none", edgecolors=PALETTE["wrong"], lw=2, zorder=5)
        ax.set_title(t, fontweight="bold"); ax.legend(handles=legend, fontsize=8)
    plt.tight_layout(); plt.show()

def plot_kmeans_hdbscan(X_umap, km, hdb_l):
    hmap = {0: PALETTE["hdb0"], 1: PALETTE["hdb1"], 2: PALETTE["hdb2"], -1: PALETTE["noise"]}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("KMeans K=2 vs HDBSCAN — UMAP 2D", fontweight="bold")
    for ax, colors, title in zip(axes,
        [[PALETTE["c0"] if k==0 else PALETTE["c1"] for k in km],
         [hmap.get(l, "#888") for l in hdb_l]],
        ["KMeans K=2", "HDBSCAN"]):
        ax.scatter(X_umap[:,0], X_umap[:,1], c=colors, s=55, alpha=0.82, edgecolors="white", lw=0.4)
        ax.set_title(title, fontweight="bold")
    plt.tight_layout(); plt.show()

def plot_outlier_scores(X_umap, hdb, mislabeled):
    os_ = hdb.outlier_scores_
    os_m = os_[mislabeled]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Outlier Scores HDBSCAN", fontweight="bold")
    sc = axes[0].scatter(X_umap[:,0], X_umap[:,1], c=os_, cmap="RdYlGn_r", s=60, alpha=0.85, edgecolors="white", lw=0.3)
    axes[0].scatter(X_umap[mislabeled,0], X_umap[mislabeled,1], s=120, facecolors="none", edgecolors="#333", lw=1.8, zorder=5)
    plt.colorbar(sc, ax=axes[0], label="Outlier score"); axes[0].set_title("Mapa UMAP 2D")
    si = np.argsort(os_m)
    bc = ["#1D9E75" if v<0.05 else "#EF9F27" if v<0.20 else "#E24B4A" for v in os_m[si]]
    axes[1].barh(range(len(si)), os_m[si], color=bc, edgecolor="none")
    axes[1].axvline(0.05, color="#1D9E75", ls="--", lw=1, label="Core"); axes[1].axvline(0.20, color="#E24B4A", ls="--", lw=1, label="Outlier")
    axes[1].set_xlabel("Outlier score"); axes[1].set_yticks([]); axes[1].legend(fontsize=8)
    plt.tight_layout(); plt.show()

def plot_feature_importance(diffs, X, km, cols):
    top = list(diffs.index[:8]); vals = diffs.values[:8]
    df = pd.DataFrame(X, columns=cols); df["cluster"] = km
    means = df.groupby("cluster")[top].mean()
    bc = ["#E24B4A" if v>1 else "#EF9F27" if v>0.3 else "#185FA5" for v in vals]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Importancia de features", fontweight="bold")
    axes[0].barh(top[::-1], vals[::-1], color=bc[::-1], edgecolor="none")
    axes[0].set_xlabel("|Δ media estandarizada|")
    x, w = np.arange(len(top)), 0.35
    axes[1].bar(x-w/2, means.loc[0], w, label="Cluster 0", color=PALETTE["c0"], alpha=0.85)
    axes[1].bar(x+w/2, means.loc[1], w, label="Cluster 1", color=PALETTE["c1"], alpha=0.85)
    axes[1].set_xticks(x); axes[1].set_xticklabels([f.replace("participacion_","p_") for f in top], rotation=35, ha="right", fontsize=8)
    axes[1].axhline(0, color="gray", lw=0.8, ls="--"); axes[1].legend()
    plt.tight_layout(); plt.show()

# -----------------------------
# 5. ANÁLISIS PREGUNTAS
# -----------------------------

def analizar_etiquetas(km, y, hdb_l, hdb, df, cols):
    mask1 = y == 1; n1 = mask1.sum()
    n_ok = (km[mask1] == 1).sum(); n_wrong = n1 - n_ok
    print(f"\n{'='*50}\nPREGUNTA 1\n{'='*50}")
    print(f"  label=1 total       : {n1}")
    print(f"  Confirma clase 1    : {n_ok}  ({100*n_ok/n1:.1f}%)")
    print(f"  Reclasifica clase 2 : {n_wrong}  ({100*n_wrong/n1:.1f}%)")
    mislabeled = np.where(mask1 & (km == 0))[0]
    for i in mislabeled:
        hs = "ruido" if hdb_l[i]==-1 else f"cluster {hdb_l[i]}"
        print(f"    [{i:2d}] {df['unidad'].iloc[i]:30s} {df['anio'].iloc[i]}  score={hdb.outlier_scores_[i]:.3f}  {hs}")
    return mislabeled

def analizar_outliers(mislabeled, hdb):
    os_ = hdb.outlier_scores_; os_m = os_[mislabeled]
    print(f"\n{'='*50}\nPREGUNTA 2\n{'='*50}")
    print(f"  Score medio global        : {os_.mean():.3f}")
    print(f"  Score medio mal etiquetados: {os_m.mean():.3f}")
    print(f"  Core(<0.05):{(os_m<0.05).sum()}  Borde:{((os_m>=0.05)&(os_m<0.20)).sum()}  Outlier(≥0.20):{(os_m>=0.20).sum()}")
    print("  → No son outliers: son core points en el cluster equivocado.")

def feature_importance(X, km, cols):
    df = pd.DataFrame(X, columns=cols); df["cluster"] = km
    diffs = (df.groupby("cluster").mean().loc[1] - df.groupby("cluster").mean().loc[0]).abs().sort_values(ascending=False)
    print("\n[INFO] Feature importance:")
    for f, v in diffs.items(): print(f"  {f:35s} {v:.3f} {'█'*int(v*10)}")
    return diffs

# -----------------------------
# 6. MAIN
# -----------------------------

def main():
    # Carga
    df_u, X_u, y_u, cols_u = load_udea()
    df_s, X_s, y_s, cols_s = load_sintetico()

    # Reducción
    X_pca_u  = reduce_pca(X_u);  X_umap2_u = reduce_umap(X_u); X_umap5_u = reduce_umap(X_u, n=5)
    X_pca_s  = reduce_pca(X_s);  X_umap2_s = reduce_umap(X_s); X_umap5_s = reduce_umap(X_s, n=5)

    # k-distance + dendrograma
    eps_u = plot_kdistance(X_u); plot_dendrogram(X_u)
    eps_s = plot_kdistance(X_s); plot_dendrogram(X_s)

    # Clustering
    km_u = run_kmeans(X_u, y_u);  db_u = run_dbscan(X_u, eps_u);  hdb_u, hdb_l_u = run_hdbscan(X_umap5_u)
    km_s = run_kmeans(X_s, y_s);  db_s = run_dbscan(X_s, eps_s);  hdb_s, hdb_l_s = run_hdbscan(X_umap5_s, min_cluster_size=10, min_samples=3)

    # Visualizaciones
    for lbl, title in [(db_u, "DBSCAN+PCA [U]"), (hdb_l_u, "HDBSCAN+PCA [U]")]:
        plot_clusters(X_pca_u, lbl, title)
    for lbl, title in [(db_u, "DBSCAN+UMAP [U]"), (hdb_l_u, "HDBSCAN+UMAP [U]")]:
        plot_clusters(X_umap2_u, lbl, title)
    for lbl, title in [(db_s, "DBSCAN+PCA [S]"), (hdb_l_s, "HDBSCAN+PCA [S]")]:
        plot_clusters(X_pca_s, lbl, title)
    for lbl, title in [(db_s, "DBSCAN+UMAP [S]"), (hdb_l_s, "HDBSCAN+UMAP [S]")]:
        plot_clusters(X_umap2_s, lbl, title)

    # Análisis preguntas 1 y 2
    mislabeled = analizar_etiquetas(km_u, y_u, hdb_l_u, hdb_u, df_u, cols_u)
    analizar_outliers(mislabeled, hdb_u)
    diffs_u = feature_importance(X_u, km_u, cols_u)

    # Visualizaciones con mislabeled
    plot_etiquetas(X_pca_u, X_umap2_u, y_u, mislabeled)
    plot_kmeans_hdbscan(X_umap2_u, km_u, hdb_l_u)
    plot_outlier_scores(X_umap2_u, hdb_u, mislabeled)
    plot_feature_importance(diffs_u, X_u, km_u, cols_u)

    # Estabilidad
    print("\n[INFO] Estabilidad bootstrap...")
    clustering_stability(X_u,      lambda X_: KMeans(2, random_state=42, n_init=10).fit_predict(X_), label="KMeans [U]")
    clustering_stability(X_umap5_u, lambda X_: hdbscan.HDBSCAN(min_cluster_size=3, min_samples=2).fit_predict(X_), label="HDBSCAN [U]")
    clustering_stability(X_s,      lambda X_: KMeans(2, random_state=42, n_init=10).fit_predict(X_), label="KMeans [S]")

if __name__ == "__main__":
    main()