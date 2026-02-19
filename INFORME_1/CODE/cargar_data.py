# cargar el dataset desde un archivo CSV

import pandas as pd
import os

def load_data(path):
    # Verificación existencia del archivo
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n[ERROR] No se encontró el archivo en: {path}"
            "\nRevisa que la ruta esté bien escrita y que el archivo sí exista."
        )

    # Evitar problemas con caracteres especiales
    df = pd.read_csv(path, encoding='utf-8')

    print(f"[OK] Datos cargados correctamente desde: {path}")
    print(f"     Filas cargadas : {df.shape[0]}")
    print(f"     Columnas cargadas: {df.shape[1]}")

    return df


def mostrar_muestra(df, n=5):
    # Muestra las primeras filas para revisar que los datos se hayan cargado correctamente
    print(f"\nPrimeras {n} filas del dataset:")
    print("-" * 60)

    print(df.head(n).to_string(index=True))

    print("-" * 60)