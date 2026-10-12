"""Construcción de variables para los modelos MiBici."""

import pandas as pd

from mibici.config import PROCESSED_DATA_DIR


def crear_demanda_horaria():
    archivos = sorted(
        PROCESSED_DATA_DIR.glob(
            "datos_abiertos_2026_*_integrado.parquet"
        )
    )

    if len(archivos) != 6:
        raise RuntimeError("Se requieren los 6 meses integrados")

    datos = []

    for archivo in archivos:
        df = pd.read_parquet(
            archivo,
            columns=["inicio_del_viaje", "origen_id"]
        )

        df["fecha_hora"] = (
            df["inicio_del_viaje"].dt.floor("h")
        )

        conteos = (
            df.groupby(["origen_id", "fecha_hora"])
            .size()
            .reset_index(name="viajes")
        )

        datos.append(conteos)
        print(f"[OK] {archivo.name}")

    demanda = pd.concat(datos, ignore_index=True)

    demanda = (
        demanda.groupby(["origen_id", "fecha_hora"], as_index=False)
        ["viajes"].sum()
    )

    estaciones = sorted(demanda["origen_id"].unique())

    horas = pd.date_range(
        start="2026-01-01 00:00:00",
        end="2026-06-30 23:00:00",
        freq="h",
    )

    indice = pd.MultiIndex.from_product(
        [estaciones, horas],
        names=["origen_id", "fecha_hora"],
    )

    demanda = (
        demanda.set_index(["origen_id", "fecha_hora"])
        .reindex(indice, fill_value=0)
        .reset_index()
    )

    demanda["hora"] = demanda["fecha_hora"].dt.hour
    demanda["dia_semana"] = demanda["fecha_hora"].dt.dayofweek
    demanda["mes"] = demanda["fecha_hora"].dt.month
    demanda["fin_semana"] = (
        demanda["dia_semana"] >= 5
    ).astype(int)

    demanda = demanda.sort_values(
        ["origen_id", "fecha_hora"]
    )

    # Objetivo: viajes de la siguiente hora.
    demanda["target_siguiente_hora"] = (
        demanda.groupby("origen_id")["viajes"]
        .shift(-1)
    )

    demanda = demanda.dropna(
        subset=["target_siguiente_hora"]
    )

    demanda["target_siguiente_hora"] = (
        demanda["target_siguiente_hora"].astype(int)
    )

    destino = (
        PROCESSED_DATA_DIR / "p1_demanda_horaria.parquet"
    )

    demanda.to_parquet(destino, index=False)

    print("\n===== RESULTADOS P1 =====")
    print("Estaciones:", len(estaciones))
    print("Horas por estación antes del shift:", len(horas))
    print("Filas generadas:", len(demanda))
    print("Archivo:", destino)


if __name__ == "__main__":
    crear_demanda_horaria()
