# analisis del dataset

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

matplotlib.use('Agg')

# Carpeta donde se guarda los histogramas
CARPETA_GRAFICOS = "outputs/graficos"


# Crea la carpeta de graficos si no existe
def _crear_carpeta_graficos():
    os.makedirs(CARPETA_GRAFICOS, exist_ok=True)


# Muestra informacion general del dataset
def informacion_basica(df):

    print("\n" + "=" * 60)
    print("FASE EDA - INSPECCION INICIAL DEL DATASET")
    print("=" * 60)

    num_filas, num_columnas = df.shape

    print("\nDIMENSIONES DEL DATASET")
    print("Filas:", num_filas)
    print("Columnas:", num_columnas)
    print("Total de celdas:", num_filas * num_columnas)

    print("\nTIPOS DE DATOS POR COLUMNA")

    tipos_df = pd.DataFrame({
        'Columna': df.columns,
        'Tipo de dato': df.dtypes.values,
        'Ejemplo': [
            str(df[col].dropna().iloc[0])
            if df[col].dropna().shape[0] > 0 else "N/A"
            for col in df.columns
        ]
    })

    print(tipos_df.to_string(index=False))

    cols_numericas = df.select_dtypes(include=['int64', 'float64']).columns
    cols_categoricas = df.select_dtypes(include=['object', 'category']).columns

    print("\nResumen de variables")
    print("Variables numericas:", len(cols_numericas))
    print("Variables categoricas:", len(cols_categoricas))

    valores_nulos = df.isnull().sum()
    total_nulos = int(valores_nulos.sum())

    print("\nVALORES NULOS")

    if total_nulos == 0:
        print("No se encontraron valores faltantes")
    else:
        print("Total de valores faltantes:", total_nulos)

        porcentaje_nulos = (valores_nulos / num_filas * 100).round(2)

        nulos_df = pd.DataFrame({
            'Columna': valores_nulos.index,
            'Valores faltantes': valores_nulos.values,
            'Porcentaje (%)': porcentaje_nulos.values
        })

        nulos_df = nulos_df[nulos_df['Valores faltantes'] > 0]
        print(nulos_df.to_string(index=False))

    print("\n" + "=" * 60 + "\n")


# Calcula estadisticas descriptivas de las columnas numericas
def estadisticas_descriptivas(df):

    print("\n" + "=" * 60)
    print("FASE EDA - ESTADISTICAS DESCRIPTIVAS")
    print("=" * 60)

    if df.empty:
        print("El DataFrame esta vacio")
        return

    columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(columnas_numericas) == 0:
        print("No se encontraron columnas numericas")
        return

    print("\nAnalizando", len(columnas_numericas), "variables numericas")
    print(", ".join(columnas_numericas))

    estadisticos = {}

    for columna in columnas_numericas:

        valores = df[columna].dropna()

        if len(valores) == 0:
            print("La columna", columna, "no tiene valores validos")
            continue

        media = valores.mean()
        mediana = valores.median()

        moda_serie = valores.mode()
        moda = float(moda_serie.iloc[0]) if len(moda_serie) > 0 else np.nan

        desviacion_std = valores.std()
        minimo = valores.min()
        maximo = valores.max()

        estadisticos[columna] = {
            'Media': round(media, 4),
            'Mediana': round(mediana, 4),
            'Moda': round(moda, 4) if not np.isnan(moda) else 'N/A',
            'Desv_Estandar': round(desviacion_std, 4),
            'Minimo': round(minimo, 4),
            'Maximo': round(maximo, 4),
        }

    resumen_df = pd.DataFrame(estadisticos).T

    print("\nTabla resumen")
    print(resumen_df.to_string())

    print("\n" + "=" * 60 + "\n")


# Genera y guarda histogramas de las columnas numericas
def graficar_histogramas(df):

    print("\n" + "=" * 60)
    print("FASE EDA - HISTOGRAMAS")
    print("=" * 60)

    _crear_carpeta_graficos()

    if df.empty:
        print("El DataFrame esta vacio")
        return

    columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(columnas_numericas) == 0:
        print("No hay variables numericas para graficar")
        return

    print("\nGenerando histogramas")

    for columna in columnas_numericas:

        valores = df[columna].dropna()

        if len(valores) == 0:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(valores, bins=20, edgecolor='black', alpha=0.75)

        media = valores.mean()
        mediana = valores.median()

        ax.axvline(media, linestyle='--', linewidth=2)
        ax.axvline(mediana, linestyle='--', linewidth=2)

        ax.set_title("Distribucion de " + columna)
        ax.set_xlabel(columna)
        ax.set_ylabel("Frecuencia")

        plt.tight_layout()

        nombre_archivo = CARPETA_GRAFICOS + "/histograma_" + columna + ".png"
        plt.savefig(nombre_archivo, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(columna, "guardado en", nombre_archivo)

    print("\n" + "=" * 60 + "\n")


# Detecta outliers usando el metodo IQR
def detectar_outliers(df):

    print("\n" + "=" * 60)
    print("FASE EDA - DETECCION DE OUTLIERS IQR")
    print("=" * 60)

    if df.empty:
        print("El DataFrame esta vacio")
        return {}

    columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(columnas_numericas) == 0:
        print("No hay columnas numericas para analizar")
        return {}

    outliers_por_columna = {}
    total_outliers = 0

    for columna in columnas_numericas:

        valores = df[columna].dropna()

        if len(valores) < 4:
            continue

        Q1 = valores.quantile(0.25)
        Q3 = valores.quantile(0.75)
        IQR = Q3 - Q1

        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR

        mascara_outliers = (
            (df[columna] < limite_inferior) |
            (df[columna] > limite_superior)
        )

        indices_outliers = df[mascara_outliers].index.tolist()

        outliers_por_columna[columna] = indices_outliers
        total_outliers += len(indices_outliers)

        print(columna + ":", len(indices_outliers), "outliers detectados")

    print("\nTotal de outliers detectados:", total_outliers)
    print("\n" + "=" * 60 + "\n")

    return outliers_por_columna