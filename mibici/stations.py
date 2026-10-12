"""Preparación de la nomenclatura de estaciones MiBici."""

import pandas as pd

from mibici.config import PROJ_ROOT, PROCESSED_DATA_DIR


def main():
    origen = (
        PROJ_ROOT
        / "data"
        / "external"
        / "nomenclatura_2026_09.csv"
    )

    estaciones = pd.read_csv(origen)

    columnas = {
        "id": "estacion_id",
        "name": "estacion_nombre",
        "latitude": "latitud",
        "longitude": "longitud",
        "dpcapacity": "capacidad",
    }

    estaciones = estaciones.rename(columns=columnas)

    estaciones["estacion_id"] = pd.to_numeric(
        estaciones["estacion_id"], errors="raise"
    ).astype("int64")

    if estaciones["estacion_id"].duplicated().any():
        raise ValueError("Existen IDs duplicados")

    estaciones["capacidad"] = pd.to_numeric(
        estaciones["capacidad"], errors="coerce"
    )

    estaciones["capacidad_valida"] = (
        estaciones["capacidad"] > 0
    )

    estaciones.loc[
        ~estaciones["capacidad_valida"], "capacidad"
    ] = float("nan")

    estaciones["latitud"] = pd.to_numeric(
        estaciones["latitud"], errors="coerce"
    )

    estaciones["longitud"] = pd.to_numeric(
        estaciones["longitud"], errors="coerce"
    )

    PROCESSED_DATA_DIR.mkdir(
        parents=True, exist_ok=True
    )

    destino = (
        PROCESSED_DATA_DIR / "estaciones_2026_09.parquet"
    )

    estaciones.to_parquet(destino, index=False)

    print("Estaciones procesadas:", len(estaciones))
    print(
        "Con capacidad válida:",
        estaciones["capacidad_valida"].sum()
    )
    print(
        "Sin capacidad válida:",
        (~estaciones["capacidad_valida"]).sum()
    )
    print("Archivo generado:", destino)


if __name__ == "__main__":
    main()
