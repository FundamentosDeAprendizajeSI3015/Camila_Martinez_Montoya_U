

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OrdinalEncoder,
    StandardScaler,
)

random_state = 42

plt.rc('font', family='serif', size=12)

# Cargamos el dataset ya limpio (generado en limpiezaDatos__1_.ipynb):
data = pd.read_csv('dataset_limpio.csv')
data

data.describe(include='all')

print(data.info())
data.select_dtypes(include=np.number).hist(bins=15, figsize=(20, 15));

# Verificamos que no haya nulos (el dataset ya fue limpiado):
print("Nulos por columna:")
print(data.isnull().sum())

# Distribución de la variable objetivo:
print("Distribución de label (0=Bajo Riesgo, 1=Alto Riesgo):")
print(data['label'].value_counts())
data['label'].value_counts().plot(kind='bar', color=['steelblue','tomato'], edgecolor='white')
plt.xticks([0,1], ['Bajo Riesgo (0)', 'Alto Riesgo (1)'], rotation=0)
plt.title('Distribución del Target')
plt.ylabel('Frecuencia')
plt.tight_layout();

# Tasa de riesgo por unidad académica:
data.groupby('unidad')['label'].mean().sort_values().plot(
    kind='barh', color='steelblue', edgecolor='white')
plt.axvline(0.5, color='tomato', linestyle='--', label='50%')
plt.title('Tasa de Riesgo por Unidad')
plt.xlabel('Proporción de periodos en riesgo')
plt.legend()
plt.tight_layout();

# Identificamos columnas categóricas y numéricas:
cat_cols = ['unidad']
num_cols = data.select_dtypes(include=np.number).columns.difference(['anio','label']).tolist()
print("Categóricas:", cat_cols)
print("Numéricas:  ", num_cols)

# OrdinalEncoder para variables categóricas (apropiado para árboles):
categorical_transformer = Pipeline(
    steps=[("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]
)
encoder = categorical_transformer.fit(data[cat_cols])
print("Categorías encontradas:", encoder["encoder"].categories_)

# Split estratificado:
X = data.drop(columns=['label', 'anio'])
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_state, stratify=y
)
print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

# Pipeline de preprocesamiento + clasificadores:
preprocessor = ColumnTransformer(
    transformers=[('cat', categorical_transformer, cat_cols)],
    remainder='passthrough'
)

rf_base = RandomForestClassifier(random_state=random_state)
gb_base = GradientBoostingClassifier(random_state=random_state)

pipeline_rf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', rf_base)])
pipeline_gb = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', gb_base)])

param_grid = {
    'classifier__n_estimators':    [50, 100],
    'classifier__max_depth':       [3, 5, None],
    'classifier__min_samples_leaf':[1, 5, 10],
}

rf = GridSearchCV(pipeline_rf, cv=5, param_grid=param_grid, scoring='f1', n_jobs=-1)
gb = GridSearchCV(pipeline_gb, cv=5, param_grid=param_grid, scoring='f1', n_jobs=-1)


rf.fit(X_train, y_train)
gb.fit(X_train, y_train)

print("Mejores parámetros — Random Forest:")
print(rf.best_params_)
print("\nMejores parámetros — Gradient Boosting:")
print(gb.best_params_)

for split_name, X_s, y_s in [('Train', X_train, y_train), ('Test', X_test, y_test)]: 
    print(f"\n{'='*50}  {split_name} set  {'='*50}")
    for model, name in [(rf, 'Random Forest'), (gb, 'Gradient Boosting')]:
        print(f"\n--- {name} ---")
        print(classification_report(y_s, model.predict(X_s),
                                    target_names=['Bajo Riesgo', 'Alto Riesgo']))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, model, name in zip(axes, [rf, gb], ['Random Forest', 'Gradient Boosting']):
    cm = confusion_matrix(y_test, model.predict(X_test))
    ConfusionMatrixDisplay(cm, display_labels=['Bajo Riesgo', 'Alto Riesgo']).plot(
        ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'Matriz de Confusión\n{name}')
plt.tight_layout();

# Importancia de características — Random Forest:
best_rf = rf.best_estimator_['classifier']
pre     = rf.best_estimator_['preprocessor']
feat_names_cat = (pre.named_transformers_['cat']['encoder']
                     .get_feature_names_out(cat_cols).tolist())
feat_names = feat_names_cat + num_cols
importances = pd.Series(best_rf.feature_importances_, index=feat_names)
importances = importances[importances > 0].sort_values()
importances.plot(kind='barh', color='steelblue', edgecolor='white', figsize=(9, 5))
plt.title('Importancia de Características — Random Forest')
plt.xlabel('Importancia (Gini)')
plt.tight_layout();

# PCA para visualizar separabilidad:
from sklearn.preprocessing import StandardScaler
X_enc    = rf.best_estimator_['preprocessor'].transform(X)
X_scaled = StandardScaler().fit_transform(X_enc)
pcs      = PCA(n_components=2, random_state=random_state).fit_transform(X_scaled)
colors   = ['steelblue' if yi == 0 else 'tomato' for yi in y]
plt.figure(figsize=(8, 5))
plt.scatter(pcs[:, 0], pcs[:, 1], c=colors, alpha=0.8, edgecolors='white')
from matplotlib.patches import Patch
plt.legend(handles=[Patch(color='steelblue', label='Bajo Riesgo (0)'),
                    Patch(color='tomato',     label='Alto Riesgo (1)')])
plt.title('PCA — Separabilidad de Clases')
plt.xlabel('PC1'); plt.ylabel('PC2')
plt.tight_layout()