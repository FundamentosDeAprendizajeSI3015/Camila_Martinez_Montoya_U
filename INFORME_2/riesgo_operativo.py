import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import skfuzzy as fuzz
from scipy.stats import mode
from sklearn.decomposition import PCA

OUTPUT_DIR = 'graficas'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. CARGA Y PREPROCESAMIENTO

df = pd.read_csv('dataset_riesgo_operativo (1).csv')

features = ['equipo', 'semana', 't_prom', 'pct_fuera', 'pct_cumplimiento', 'interaccion', 'des_carga', 'correcciones']
X = df[features].values

y_original = pd.qcut(df['riesgo_final'], q=3, labels=[0,1,2]).astype(int).values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA para graficar
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


# 2. FUNCION DE GRAFICAS

def plot_clusters(labels, title, filename):
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=labels)
    plt.title(title)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path)
    plt.close()


# 3. CLUSTERING + SILHOUETTE

results_silhouette = {}

# KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
km_labels = kmeans.fit_predict(X_scaled)
results_silhouette['KMeans'] = silhouette_score(X_scaled, km_labels)
plot_clusters(km_labels, 'KMeans', 'kmeans.png')

# Fuzzy C-Means
cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(X_scaled.T, c=3, m=2, error=0.005, maxiter=1000)
fcm_labels = np.argmax(u, axis=0)
results_silhouette['Fuzzy'] = silhouette_score(X_scaled, fcm_labels)
plot_clusters(fcm_labels, 'Fuzzy C-Means', 'fuzzy.png')

# DBSCAN
dbscan = DBSCAN(eps=1.5, min_samples=5)
db_labels = dbscan.fit_predict(X_scaled)
mask = db_labels != -1
if len(set(db_labels[mask])) > 1:
    results_silhouette['DBSCAN'] = silhouette_score(X_scaled[mask], db_labels[mask])
else:
    results_silhouette['DBSCAN'] = -1
plot_clusters(db_labels, 'DBSCAN', 'dbscan.png')

# Jerárquico
agg = AgglomerativeClustering(n_clusters=3)
agg_labels = agg.fit_predict(X_scaled)
results_silhouette['Jerarquico'] = silhouette_score(X_scaled, agg_labels)
plot_clusters(agg_labels, 'Jerarquico', 'jerarquico.png')

# Subtractive (simplificado)
sub_labels = KMeans(n_clusters=3, random_state=42).fit_predict(X_scaled)
results_silhouette['Subtractive'] = silhouette_score(X_scaled, sub_labels)
plot_clusters(sub_labels, 'Subtractive', 'subtractive.png')

print("SILHOUETTE SCORES")
for k,v in results_silhouette.items():
    print(f"{k}: {v:.4f}")

# GRAFICA SILHOUETTE
plt.figure(figsize=(8,5))
sns.barplot(x=list(results_silhouette.keys()), y=list(results_silhouette.values()))
plt.title('Comparación Silhouette Score')
plt.ylabel('Score')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'silhouette.png'))
plt.close()


# 4. REETIQUETADO

all_labels = np.vstack([km_labels, fcm_labels, db_labels, agg_labels, sub_labels])

for i in range(all_labels.shape[0]):
    if -1 in all_labels[i]:
        mask = all_labels[i] != -1
        replacement = mode(all_labels[i][mask])[0]
        all_labels[i][all_labels[i] == -1] = replacement

new_labels = mode(all_labels, axis=0)[0]

cambios = np.sum(new_labels != y_original)
pct = cambios / len(y_original) * 100
print(f"\n% cambio etiquetas: {pct:.2f}%")


# 5. GRAFICA DISTRIBUCION

plt.figure()
sns.countplot(x=y_original, alpha=0.5, label='Original')
sns.countplot(x=new_labels, alpha=0.5, label='Nuevo')
plt.legend()
plt.title('Distribución de etiquetas')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'distribucion.png'))
plt.close()


# 6. MODELOS SUPERVISADOS

X_train, X_test, y_orig_train, y_orig_test, y_new_train, y_new_test = train_test_split(
    X_scaled, y_original, new_labels, test_size=0.25, random_state=42
)

def evaluate(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, pred),
        'precision': precision_score(y_test, pred, average='weighted'),
        'recall': recall_score(y_test, pred, average='weighted'),
        'f1': f1_score(y_test, pred, average='weighted')
    }

models = {
    'DT': DecisionTreeClassifier(max_depth=5),
    'LR': LogisticRegression(max_iter=1000)
}

print("RESULTADOS SUPERVISADOS")

for name, model in models.items():
    orig = evaluate(model, X_train, y_orig_train, X_test, y_orig_test)
    new = evaluate(model, X_train, y_new_train, X_test, y_new_test)

    print(f"{name}")
    print("Original:", orig)
    print("Nuevo:", new)

# ARBOL DE DECISION
from sklearn.tree import plot_tree

model_tree = DecisionTreeClassifier(max_depth=5)
model_tree.fit(X_train, y_new_train)

plt.figure(figsize=(15,10))
plot_tree(model_tree, feature_names=features, class_names=['Bajo','Medio','Alto'], filled=True)
plt.title('Árbol de decisión')
plt.savefig(os.path.join(OUTPUT_DIR, 'arbol.png'))
plt.close()

# IMPORTANCIA VARIABLES
plt.figure(figsize=(8,5))
sns.barplot(x=model_tree.feature_importances_, y=features)
plt.title('Importancia de variables')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'importancia.png'))
plt.close()

# COMPARACION METRICAS
metrics_orig = evaluate(DecisionTreeClassifier(max_depth=5), X_train, y_orig_train, X_test, y_orig_test)
metrics_new = evaluate(DecisionTreeClassifier(max_depth=5), X_train, y_new_train, X_test, y_new_test)

import pandas as pd

df_metrics = pd.DataFrame([metrics_orig, metrics_new], index=['Original','Nuevo'])

df_metrics.plot(kind='bar')
plt.title('Comparación métricas')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'metricas.png'))
plt.close()


# 7. REGRESION LINEAL

lin = LinearRegression()
lin.fit(X_train, y_orig_train)
pred = lin.predict(X_test)
print("\nLinear Reg (orig):", mean_squared_error(y_orig_test, pred), r2_score(y_orig_test, pred))


# 8. CROSS VALIDATION

print("\n--- CROSS VALIDATION (Decision Tree) ---")
cv_scores = cross_val_score(DecisionTreeClassifier(max_depth=5), X_scaled, y_original, cv=5)
print("CV Accuracy:", cv_scores.mean())