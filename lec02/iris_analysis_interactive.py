# iris_analysis_interactive.py
# -*- coding: utf-8 -*-
"""
Análisis completo del dataset Iris con visualizaciones interactivas.
Incluye: EDA, ingeniería de características, modelado clásico y gráficos interactivos
(exportados a HTML) con Plotly.

Ejecución:
    python iris_analysis_interactive.py

Requisitos (instalación sugerida si falta alguno):
    pandas, numpy, matplotlib, seaborn, scikit-learn, joblib, plotly
"""
from __future__ import annotations
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from joblib import dump

# --- Visualizaciones interactivas ---
import plotly.express as px
import plotly.graph_objects as go
from plotly.io import write_html

# Fijar semilla aleatoria para que los resultados sean siempre los mismos
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ----------------------------------
# 1 Carga de datos
# ----------------------------------
iris = datasets.load_iris()
# X son las caracteristicas (features) y y son las etiquetas (target)
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="target")
class_names = iris.target_names

# Renombrar columnas a formato limpio
X.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in X.columns]

# Crear un DataFrame completo con las especies para hacer EDA mas facil
species_map = {i: name for i, name in enumerate(class_names)}
df = pd.concat([X, y.map(species_map).rename("species")], axis=1)

# Carpeta de salidas
os.makedirs("outputs", exist_ok=True)

# ----------------------------------
# 2 EDA basica (texto y PNG)
# ----------------------------------
# Exploracion inicial de los datos
print("\nDimensiones:", df.shape)
print("\nPrimeras filas:\n", df.head())
print("\nDescripcion estadistica:\n", df.describe(include="all"))
print("\nClases:\n", df["species"].value_counts())

# Matriz de correlacion para ver relaciones entre variables
corr = df.drop(columns=["species"]).corr()
plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de correlacion - Iris")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png", dpi=150)
plt.close()

# Pairplot para ver todas las relaciones entre pares de variables
sns.pairplot(df, hue="species", corner=True, palette="Set2")
plt.suptitle("Pairplot Iris", y=1.02)
plt.savefig("outputs/pairplot.png", dpi=150)
plt.close()

# ----------------------------------
# 3 Visualizaciones interactivas (HTML)
# ----------------------------------
# Graficos interactivos con Plotly que se pueden abrir en el navegador

# a) Scatter matrix interactivo - muestra todas las relaciones entre variables
fig_scatter_matrix = px.scatter_matrix(
    df,
    dimensions=X.columns,
    color="species",
    title="Iris - Scatter Matrix Interactivo",
    height=800,
    color_discrete_sequence=px.colors.qualitative.Set2
)
write_html(fig_scatter_matrix, file="outputs/interactive_scatter_matrix.html", include_plotlyjs="cdn")

# b) Coordenadas paralelas - util para comparar multiples variables a la vez
# Necesito normalizar los datos primero para que todas las escalas sean comparables
X_norm = (X - X.min())/(X.max() - X.min())
X_norm["species"] = df["species"]
fig_parallel = px.parallel_coordinates(
    X_norm,
    dimensions=[c for c in X.columns],
    color="species",
    color_continuous_scale=px.colors.qualitative.Pastel,
    title="Iris - Coordenadas Paralelas Interactivo"
)
write_html(fig_parallel, file="outputs/interactive_parallel_coordinates.html", include_plotlyjs="cdn")

# c) Grafico 3D usando las 3 caracteristicas mas importantes
fig_3d = px.scatter_3d(
    df,
    x="petal_length", y="petal_width", z="sepal_length",
    color="species",
    title="Iris - Dispersion 3D Interactivo", height=600,
    color_discrete_sequence=px.colors.qualitative.Set2
)
write_html(fig_3d, file="outputs/interactive_scatter_3d.html", include_plotlyjs="cdn")

# d) PCA - reduce la dimensionalidad a 2 componentes principales
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(StandardScaler().fit_transform(X))
fig_pca = px.scatter(
    x=X_pca[:,0], y=X_pca[:,1], color=df["species"],
    labels={"x": "PC1", "y": "PC2"}, title="Iris - PCA 2D Interactivo",
    color_discrete_sequence=px.colors.qualitative.Set2
)
write_html(fig_pca, file="outputs/interactive_pca.html", include_plotlyjs="cdn")

# e) t-SNE - otra tecnica de reduccion de dimensionalidad (mas lento pero mejor separacion)
tsne = TSNE(n_components=2, random_state=RANDOM_STATE, init="pca", learning_rate="auto")
X_tsne = tsne.fit_transform(StandardScaler().fit_transform(X))
fig_tsne = px.scatter(
    x=X_tsne[:,0], y=X_tsne[:,1], color=df["species"],
    labels={"x": "t-SNE 1", "y": "t-SNE 2"}, title="Iris - t-SNE Interactivo",
    color_discrete_sequence=px.colors.qualitative.Set2
)
write_html(fig_tsne, file="outputs/interactive_tsne.html", include_plotlyjs="cdn")

# ----------------------------------
# 4 Ingenieria de caracteristicas
# ----------------------------------
# Crear nuevas caracteristicas que pueden ayudar al modelo
# Agrego razones y areas que pueden ser mas informativas que las medidas individuales
X_feat = X.copy()
X_feat["sepal_ratio"] = X["sepal_length"]/X["sepal_width"]
X_feat["petal_ratio"] = X["petal_length"]/X["petal_width"]
X_feat["sepal_area"] = X["sepal_length"]*X["sepal_width"]
X_feat["petal_area"] = X["petal_length"]*X["petal_width"]

# ----------------------------------
# 5 Particion y modelado
# ----------------------------------
# Dividir los datos en entrenamiento (75%) y prueba (25%)
# stratify=y mantiene la proporcion de clases en ambos conjuntos
X_train, X_test, y_train, y_test = train_test_split(
    X_feat, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)

# Definir varios modelos para comparar su rendimiento
# Uso Pipeline para combinar escalado y clasificador en un solo paso
models = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, multi_class="auto", random_state=RANDOM_STATE))
    ]),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier())
    ]),
    "DecisionTree": Pipeline([
        ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE))
    ]),
    "RandomForest": Pipeline([
        ("clf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE))
    ]),
    "SVM_RBF": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))
    ])
}

# Validacion cruzada con 5 folds para evaluar cada modelo
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
print("\nValidacion cruzada (accuracy media +- std):")
cv_summary = []
for name, pipe in models.items():
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
    cv_summary.append({"modelo": name, "accuracy_mean": scores.mean(), "accuracy_std": scores.std()})
    print(f"{name:>15}: {scores.mean():.3f} +- {scores.std():.3f}")
cv_df = pd.DataFrame(cv_summary)

# Guardar tabla interactiva con los resultados de validacion cruzada
fig_cv = go.Figure(data=[go.Table(
    header=dict(values=list(cv_df.columns), fill_color='lightblue', align='left'),
    cells=dict(values=[cv_df[c] for c in cv_df.columns], align='left')
)])
fig_cv.update_layout(title_text="Resultados de validacion cruzada interactivo")
write_html(fig_cv, file="outputs/interactive_cv_results.html", include_plotlyjs="cdn")

# Ajuste de hiperparametros para los dos mejores modelos (SVM y RandomForest)
param_grid_svm = {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.1, 0.01, 0.001]}
svm_grid = GridSearchCV(models["SVM_RBF"], param_grid_svm, cv=cv, scoring="accuracy", n_jobs=-1)
svm_grid.fit(X_train, y_train)

param_grid_rf = {"clf__n_estimators": [100, 200, 400], "clf__max_depth": [None, 3, 5, 7]}
rf_grid = GridSearchCV(models["RandomForest"], param_grid_rf, cv=cv, scoring="accuracy", n_jobs=-1)
rf_grid.fit(X_train, y_train)

# Seleccionar el mejor modelo entre SVM y RandomForest
best_estimator = svm_grid if svm_grid.best_score_ >= rf_grid.best_score_ else rf_grid
best_model = best_estimator.best_estimator_
best_name = "SVM_RBF" if best_estimator is svm_grid else "RandomForest"
print(f"\nModelo seleccionado: {best_name}")

# ----------------------------------
# 6 Evaluacion en test
# ----------------------------------
# Evaluar el mejor modelo en el conjunto de prueba
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
cm = confusion_matrix(y_test, y_pred)

print("\nResultados en test:")
print(f"Accuracy: {acc:.3f}")
print(f"Precision: {prec:.3f}  Recall: {rec:.3f}  F1: {f1:.3f}")
print("\nReporte de clasificacion:\n", classification_report(y_test, y_pred, target_names=class_names))

# Matriz de confusion interactiva - muestra cuantas predicciones fueron correctas/incorrectas
fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='YlOrRd',
                   labels=dict(x="Prediccion", y="Real", color="Cuenta"),
                   x=class_names, y=class_names,
                   title=f"Matriz de confusion - {best_name} Interactivo")
write_html(fig_cm, file="outputs/interactive_confusion_matrix.html", include_plotlyjs="cdn")

# ROC-AUC - metrica adicional para evaluar el modelo (solo si el modelo puede dar probabilidades)
try:
    y_score = best_model.predict_proba(X_test)
    y_bin = pd.get_dummies(y_test).values
    auc = roc_auc_score(y_bin, y_score, multi_class='ovr', average='macro')
    print(f"ROC-AUC (macro, OvR): {auc:.3f}")
except Exception as e:
    print("No se pudo calcular ROC-AUC:", e)

# Ver que caracteristicas son mas importantes para el modelo
feature_importances = None
feature_names = X_feat.columns
try:
    clf = best_model.named_steps.get('clf')
    if hasattr(clf, 'feature_importances_'):
        feature_importances = clf.feature_importances_
    elif hasattr(clf, 'coef_'):
        # Para modelos lineales, uso los coeficientes como proxy de importancia
        coefs = clf.coef_
        feature_importances = np.mean(np.abs(coefs), axis=0)
except Exception:
    feature_importances = None

if feature_importances is not None:
    imp = pd.Series(feature_importances, index=feature_names).sort_values(ascending=False)
    fig_imp = px.bar(imp, x=imp.values, y=imp.index, orientation='h',
                     labels={'x':'Importancia relativa', 'index':'Caracteristica'},
                     title=f"Importancia de caracteristicas - {best_name} Interactivo",
                     color_discrete_sequence=['#4ECDC4'])
    write_html(fig_imp, file="outputs/interactive_feature_importance.html", include_plotlyjs="cdn")

# Guardar el modelo entrenado y un resumen de los resultados
dump(best_model, "outputs/iris_best_model.joblib")
with open("outputs/summary.txt", "w", encoding="utf-8") as f:
    f.write("Resultados del analisis Iris interactivo\n")
    f.write(f"Modelo: {best_name}\n")
    f.write(f"Accuracy test: {acc:.3f}\n")
    f.write(f"Precision: {prec:.3f}  Recall: {rec:.3f}  F1: {f1:.3f}\n")

print("\nArchivos generados en outputs/:\n - PNGs estaticos\n - interactive_*.html (abrir en navegador)\n - iris_best_model.joblib y summary.txt\n")
