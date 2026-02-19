# Importaciones de los modulos del proyecto
from cargar_data import load_data, mostrar_muestra
from eda import (
    informacion_basica,
    estadisticas_descriptivas,
    graficar_histogramas,
    detectar_outliers,
)

def main():
    # Funcion principal que ejecuta el pipeline completo de EDA

    print("\n" + "=" * 70)
    print("PIPELINE DE EXPLORACION DE DATOS EDA - RIESGO OPERATIVO")
    print("Fase: Recoleccion + Exploracion de Datos")
    print("Ciclo de vida ML - Ingenieria de Sistemas")
    print("=" * 70)

    # Cargar el dataset
    print("\n" + "-" * 70)
    print("ETAPA 1 - RECOLECCION DE DATOS")
    print("-" * 70)

    RUTA_DATASET = "../data/raw/dataset_riesgo_operativo.csv"

    df = load_data(RUTA_DATASET)

    mostrar_muestra(df, n=5)

    # Revisión inicial del dataset
    print("\n" + "-" * 70)
    print("ETAPA 2 - INSPECCION INICIAL DEL DATASET")
    print("-" * 70)

    informacion_basica(df)

    # Estadisticas descriptivas
    print("\n" + "-" * 70)
    print("ETAPA 3 - ESTADISTICAS DESCRIPTIVAS")
    print("-" * 70)

    estadisticas_descriptivas(df)

    # Generación de histogramas
    print("\n" + "-" * 70)
    print("ETAPA 4 - VISUALIZACION HISTOGRAMAS")
    print("-" * 70)

    graficar_histogramas(df)

    # Deteccion de outliers con IQR
    print("\n" + "-" * 70)
    print("ETAPA 5 - DETECCION DE OUTLIERS IQR")
    print("-" * 70)

    outliers = detectar_outliers(df)

    # Resumen final
    total_outliers = sum(len(idx) for idx in outliers.values())
    cols_con_outliers = [c for c, idx in outliers.items() if len(idx) > 0]
    cols_numericas_cnt = len(df.select_dtypes(include=['int64', 'float64']).columns)

    print("\n" + "=" * 70)
    print("PIPELINE EDA COMPLETADO - RESUMEN")
    print("=" * 70)
    print(f"\n  Dataset analizado  : {RUTA_DATASET}")
    print(f"Filas              : {df.shape[0]}")
    print(f"Columnas totales   : {df.shape[1]}")
    print(f"Variables numericas: {cols_numericas_cnt}")
    print(f"Valores nulos      : {int(df.isnull().sum().sum())}")
    print(f"Outliers detectados: {total_outliers}")

    if cols_con_outliers:
        print(f"Columnas con outliers: {', '.join(cols_con_outliers)}")

    print(f"\n  Graficos guardados en: outputs/graficos/")

    print("\n" + "=" * 70 + "\n")

    return df

# Ejecuta main
if __name__ == "__main__":
    df_resultado = main()