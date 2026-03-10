
import numpy as np
import pandas as pd
from scipy.stats import reciprocal
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

random_state = 42
np.random.seed(random_state)

plt.rc('font', family='serif', size=12)

# Cargamos el dataset limpio (generado en limpiezaDatos__1_.ipynb):
df = pd.read_csv('dataset_limpio.csv')
print(f"Shape: {df.shape}")
df.head()

# Verificación: no deben existir nulos
print("Nulos:", df.isnull().sum().sum())

# Definimos features y target:
TARGET = 'ingresos_totales'
cat_cols = ['unidad']
num_cols = [c for c in df.columns if c not in ['anio','label','unidad', TARGET]]

X = df.drop(columns=['anio','label', TARGET])
y = df[TARGET]
print("Features numéricas:", num_cols)
print("Features categóricas:", cat_cols)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_state
)
print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

# Distribución del target:
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].hist(y / 1e9, bins=15, color='steelblue', edgecolor='white')
axes[0].set_title('Distribución de Ingresos Totales')
axes[0].set_xlabel('Ingresos (Miles de millones COP)')
axes[1].scatter(range(len(y)), y.sort_values().values / 1e9, color='steelblue', alpha=0.7)
axes[1].set_title('Ingresos Totales ordenados')
axes[1].set_xlabel('Índice')
plt.tight_layout();

# Pipelines Ridge y LASSO:
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
])
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols),
])

ridge_base = Pipeline([
    ('preprocessor', preprocessor),
    ('poly',      PolynomialFeatures(include_bias=False)),
    ('scaler2',   StandardScaler()),
    ('regressor', Ridge()),
])
lasso_base = Pipeline([
    ('preprocessor', preprocessor),
    ('poly',      PolynomialFeatures(include_bias=False)),
    ('scaler2',   StandardScaler()),
    ('regressor', Lasso(max_iter=10000)),
])

param_distributions = {
    'poly__degree':     list(range(1, 4)),
    'regressor__alpha': reciprocal(1e-3, 1e3),
}

ridge = RandomizedSearchCV(ridge_base, cv=5, param_distributions=param_distributions,
                           n_iter=100, random_state=random_state, scoring='r2')
lasso = RandomizedSearchCV(lasso_base, cv=5, param_distributions=param_distributions,
                           n_iter=100, random_state=random_state, scoring='r2')

ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)

ridge.best_params_

lasso.best_params_

print('Modelo Ridge')
print(f'R²:  {ridge.score(X_test, y_test):.4f}')
print(f'MAE: {mean_absolute_error(y_test, ridge.predict(X_test))/1e9:.2f} miles de millones COP')

print('Modelo LASSO')
print(f'R²:  {lasso.score(X_test, y_test):.4f}')
print(f'MAE: {mean_absolute_error(y_test, lasso.predict(X_test))/1e9:.2f} miles de millones COP')

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, model, name in zip(axes, [ridge, lasso], ['Ridge', 'LASSO']):
    y_pred = model.predict(X_test)
    ax.scatter(y_test/1e9, y_pred/1e9, color='steelblue', alpha=0.8, edgecolors='white')
    lim = [min(y_test.min(), y_pred.min())/1e9, max(y_test.max(), y_pred.max())/1e9]
    ax.plot(lim, lim, 'r--', lw=1.5, label='Predicción perfecta')
    ax.set_xlabel('Real (miles de millones COP)')
    ax.set_ylabel('Predicho (miles de millones COP)')
    ax.set_title(f'{name}  —  R²={model.score(X_test, y_test):.3f}')
    ax.legend()
plt.tight_layout();

print('Modelo Ridge')
print(f"Intercepto: {ridge.best_estimator_['regressor'].intercept_:.2e}")
print(f"Núm. coeficientes: {len(ridge.best_estimator_['regressor'].coef_)}")

print('Modelo LASSO')
lasso_coef = lasso.best_estimator_['regressor'].coef_
print(f"Coeficientes no nulos: {np.sum(lasso_coef != 0)} / {len(lasso_coef)}")
print(f"Intercepto: {lasso.best_estimator_['regressor'].intercept_:.2e}")