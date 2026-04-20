import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, r2_score

from scipy.stats import mode


# carpeta para guardar graficas
OUTPUT_DIR = 'graficas'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# CARGA DEL DATASET
df = pd.read_csv('dataset_riesgo_operativo (1).csv')


# EDA
print(df.head())
print(df.info())
print(df.describe())

plt.figure()
sns.histplot(df['riesgo_final'])
plt.title('Distribución riesgo_final')
plt.savefig(os.path.join(OUTPUT_DIR, 'hist_riesgo.png'))
plt.close()

plt.figure(figsize=(10,5))
sns.boxplot(data=df)
plt.xticks(rotation=45)
plt.title('Boxplot variables')
plt.savefig(os.path.join(OUTPUT_DIR, 'boxplot_variables.png'))
plt.close()


# COPIA
df_clean = df.copy()


# LIMPIEZA
df_clean.columns = df_clean.columns.str.lower().str.strip()
print(df_clean.isnull().sum())


# FEATURES Y ESCALADO
features = ['equipo','semana','t_prom','pct_fuera','pct_cumplimiento','interaccion','des_carga','correcciones']

X = df_clean[features].values

y_original = pd.qcut(df_clean['riesgo_final'], q=3, labels=[0,1,2]).astype(int).values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# SPLIT 60/20/20
X_temp, X_test, y_temp, y_test = train_test_split(
    X_scaled, y_original, test_size=0.2, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42
)


# CLUSTERING SOLO TRAIN
print("Probando diferentes k para ver cual funciona mejor...")

inertia = []
k_range = range(1,10)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_train)
    inertia.append(km.inertia_)

plt.figure()
plt.plot(k_range, inertia)
plt.title('Metodo del codo (train)')
plt.xlabel('k')
plt.ylabel('Inercia')
plt.savefig(os.path.join(OUTPUT_DIR, 'codo_kmeans.png'))
plt.close()

# KMeans final
kmeans = KMeans(n_clusters=3, random_state=42)
train_clusters = kmeans.fit_predict(X_train)

print("Silhouette (train):", silhouette_score(X_train, train_clusters))


# ETIQUETAS NUEVAS
mapping = {}

for cluster in np.unique(train_clusters):
    idx = np.where(train_clusters == cluster)
    values, counts = np.unique(y_train[idx], return_counts=True)
    mapping[cluster] = values[np.argmax(counts)]

y_train_new = np.array([mapping[c] for c in train_clusters])

test_clusters = kmeans.predict(X_test)
y_test_new = np.array([mapping[c] for c in test_clusters])


# MODELOS
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
    'LR': LogisticRegression(max_iter=1000),
    'RF': RandomForestClassifier(),
    'SVM': SVC()
}

print("\n--- RESULTADOS ---")

for name, model in models.items():
    orig = evaluate(model, X_train, y_train, X_test, y_test)
    new = evaluate(model, X_train, y_train_new, X_test, y_test_new)

    print(f"\n{name}")
    print("Original:", orig)
    print("Nuevo:", new)


# ÁRBOL
model_tree = DecisionTreeClassifier(max_depth=5)
model_tree.fit(X_train, y_train_new)

plt.figure(figsize=(15,10))
plot_tree(model_tree, feature_names=features, class_names=['Bajo','Medio','Alto'], filled=True)
plt.title('Árbol de decisión')
plt.savefig(os.path.join(OUTPUT_DIR, 'arbol_decision.png'))
plt.close()

plt.figure()
sns.barplot(x=model_tree.feature_importances_, y=features)
plt.title('Importancia de variables')
plt.savefig(os.path.join(OUTPUT_DIR, 'importancia_variables.png'))
plt.close()


# REGRESIÓN
lin = LinearRegression()
lin.fit(X_train, y_train)
pred = lin.predict(X_test)

print("\nLinear Reg:", mean_squared_error(y_test, pred), r2_score(y_test, pred))


# CROSS VALIDATION
cv_scores = cross_val_score(DecisionTreeClassifier(max_depth=5), X_scaled, y_original, cv=5)
print("\nCV Accuracy:", cv_scores.mean())