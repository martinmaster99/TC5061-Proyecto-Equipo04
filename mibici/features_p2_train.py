"""Variables predictivas de P2 para la siguiente hora."""

import numpy as np
import pandas as pd
from mibici.config import PROCESSED_DATA_DIR

origen = PROCESSED_DATA_DIR / "p2_flujos_capacidad.parquet"
df = pd.read_parquet(origen)

estaciones = df[["estacion_id", "capacidad"]].drop_duplicates()

if estaciones["estacion_id"].duplicated().any():
    raise ValueError("Capacidades duplicadas por estación")

horas = pd.date_range(
    "2026-01-01 00:00:00",
    "2026-06-30 23:00:00",
    freq="h"
)

indice = pd.MultiIndex.from_product(
    [estaciones["estacion_id"].sort_values(), horas],
    names=["estacion_id", "fecha_hora"]
)

df = (
    df.set_index(["estacion_id", "fecha_hora"])
    .reindex(indice)
    .reset_index()
)

df = df.drop(columns=["capacidad", "estacion_nombre",
                       "flujo_neto"], errors="ignore")

df = df.merge(
    estaciones,
    on="estacion_id",
    how="left",
    validate="many_to_one"
)

df[["entradas", "salidas"]] = (
    df[["entradas", "salidas"]].fillna(0).astype("int32")
)

df["indice_flujo"] = (
    (df["entradas"] - df["salidas"]) / df["capacidad"]
)

df["riesgo"] = np.select(
    [
        df["indice_flujo"] <= -0.5,
        df["indice_flujo"] >= 0.5
    ],
    [1, 2],
    default=0
).astype("int8")

df = df.sort_values(["estacion_id", "fecha_hora"])

grupo = df.groupby("estacion_id", sort=False)

df["entradas_hace_1h"] = grupo["entradas"].shift(1)
df["salidas_hace_1h"] = grupo["salidas"].shift(1)
df["indice_hace_1h"] = grupo["indice_flujo"].shift(1)

# Objetivo: riesgo durante la siguiente hora del reloj.
df["target_siguiente_hora"] = grupo["riesgo"].shift(-1)

# Variables de calendario de la hora que queremos predecir.
hora_futura = df["fecha_hora"] + pd.Timedelta(hours=1)
df["hora_objetivo"] = hora_futura.dt.hour
df["dia_semana_objetivo"] = hora_futura.dt.dayofweek
df["mes_objetivo"] = hora_futura.dt.month

df = df.dropna(subset=[
    "entradas_hace_1h",
    "salidas_hace_1h",
    "indice_hace_1h",
    "target_siguiente_hora"
]).copy()

df["target_siguiente_hora"] = (
    df["target_siguiente_hora"].astype("int8")
)

destino = PROCESSED_DATA_DIR / "p2_features_final.parquet"
df.to_parquet(destino, index=False)

print("===== VARIABLES P2 =====")
print("Estaciones:", df["estacion_id"].nunique())
print("Registros:", len(df))
print("Valores nulos:", df.isna().sum().sum())

print("\n===== CLASES DEL OBJETIVO =====")
print(df["target_siguiente_hora"].value_counts().sort_index())

print("\n===== DIVISIÓN TEMPORAL =====")
train = df[df["fecha_hora"] < "2026-04-30 23:00:00"]
val = df[
    (df["fecha_hora"] >= "2026-05-01") &
    (df["fecha_hora"] < "2026-05-31 23:00:00")
]
test = df[df["fecha_hora"] >= "2026-06-01"]

print("Entrenamiento:", len(train))
print("Validación:", len(val))
print("Prueba:", len(test))
print("Archivo:", destino)
