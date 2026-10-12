"""Integra los viajes MiBici con la nomenclatura de estaciones."""

from pathlib import Path

import pandas as pd

from mibici.config import PROCESSED_DATA_DIR


def main():
    estaciones = pd.read_parquet(
        PROCESSED_DATA_DIR / "estaciones_2026_09.parquet"
    )

    columnas_estacion = [
        "estacion_id",
        "estacion_nombre",
        "latitud",
        "longitud",
        "capacidad",
        "capacidad_valida",
    ]

    estaciones = estaciones[columnas_estacion]

    archivos = sorted(
        PROCESSED_DATA_DIR.glob("datos_abiertos_2026_*.parquet")
    )

    if len(archivos) != 6:
        raise RuntimeError(
            f"Se esperaban 6 archivos y se encontraron {len(archivos)}"
        )

    total_original = 0
    total_integrado = 0

    for archivo in archivos:
        print(f"\n[INTEGRANDO] {archivo.name}")

        viajes = pd.read_parquet(archivo)
        filas_originales = len(viajes)

        for tipo in ["origen", "destino"]:
            viajes[f"{tipo}_id"] = pd.to_numeric(
                viajes[f"{tipo}_id"], errors="coerce"
            ).astype("Int64")

            datos_estacion = estaciones.rename(
                columns={
                    "estacion_id": f"{tipo}_id",
                    "estacion_nombre": f"{tipo}_nombre",
                    "latitud": f"{tipo}_latitud",
                    "longitud": f"{tipo}_longitud",
                    "capacidad": f"{tipo}_capacidad",
                    "capacidad_valida": f"{tipo}_capacidad_valida",
                }
            )

            viajes = viajes.merge(
                datos_estacion,
                on=f"{tipo}_id",
                how="left",
                validate="many_to_one",
            )

        if len(viajes) != filas_originales:
            raise RuntimeError(
                f"Se alteró el número de viajes en {archivo.name}"
            )

        destino = PROCESSED_DATA_DIR / (
            archivo.stem + "_integrado.parquet"
        )

        viajes.to_parquet(destino, index=False)

        total_original += filas_originales
        total_integrado += len(viajes)

        print(f"[OK] {len(viajes):,} viajes conservados")
        print(
            "Sin información de origen:",
            viajes["origen_nombre"].isna().sum()
        )
        print(
            "Sin información de destino:",
            viajes["destino_nombre"].isna().sum()
        )

    print("\n===== RESUMEN =====")
    print("Viajes originales:", total_original)
    print("Viajes integrados:", total_integrado)
    print("Registros perdidos:", total_original - total_integrado)


if __name__ == "__main__":
    main()
