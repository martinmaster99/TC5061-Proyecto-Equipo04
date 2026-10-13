import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

ruta = Path("data/processed/p2_features_final.parquet")
df = pd.read_parquet(ruta)

df["fecha_objetivo"] = (
    df["fecha_hora"] + pd.Timedelta(hours=1)
)

train = df[df["fecha_objetivo"] < "2026-05-01"]
val = df[
    (df["fecha_objetivo"] >= "2026-05-01") &
    (df["fecha_objetivo"] < "2026-06-01")
]
test = df[
    (df["fecha_objetivo"] >= "2026-06-01") &
    (df["fecha_objetivo"] < "2026-07-01")
]

variables = [
    "entradas_hace_1h",
    "salidas_hace_1h",
    "indice_hace_1h",
    "hora_objetivo",
    "dia_semana_objetivo",
    "mes_objetivo",
    "capacidad",
]

modelo = DummyClassifier(strategy="most_frequent")
modelo.fit(train[variables], train["target_siguiente_hora"])

mlflow.set_experiment("mibici-equipo04")

with mlflow.start_run(run_name="P2_baseline_clase_mayoritaria"):

    mlflow.log_param("pregunta", "P2")
    mlflow.log_param("modelo", "DummyClassifier")
    mlflow.log_param("estrategia", "most_frequent")
    mlflow.log_param("datos", "MiBici_2026_enero_junio")
    mlflow.log_param("version_features", "p2_features_final_v1")
    mlflow.log_param("train_periodo", "2026-01_a_2026-04")
    mlflow.log_param("val_periodo", "2026-05")
    mlflow.log_param("test_periodo", "2026-06")

    for nombre, datos in [
        ("train", train),
        ("val", val),
        ("test", test),
    ]:
        y_real = datos["target_siguiente_hora"]
        y_pred = modelo.predict(datos[variables])

        metricas = {
            "accuracy": accuracy_score(y_real, y_pred),
            "balanced_accuracy": balanced_accuracy_score(
                y_real, y_pred
            ),
            "f1_macro": f1_score(
                y_real, y_pred,
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            ),
        }

        for metrica, valor in metricas.items():
            mlflow.log_metric(f"{nombre}_{metrica}", valor)

        print(f"\n===== {nombre.upper()} =====")
        for metrica, valor in metricas.items():
            print(f"{metrica}: {valor:.4f}")

        print(
            classification_report(
                y_real,
                y_pred,
                labels=[0, 1, 2],
                zero_division=0,
            )
        )

        if nombre == "test":
            matriz = confusion_matrix(
                y_real, y_pred, labels=[0, 1, 2]
            )

            carpeta = Path("reports/p2")
            carpeta.mkdir(parents=True, exist_ok=True)

            np.savetxt(
                carpeta / "matriz_confusion_baseline.csv",
                matriz,
                delimiter=",",
                fmt="%d",
            )

            with open(
                carpeta / "metricas_baseline.json",
                "w",
                encoding="utf-8",
            ) as archivo:
                json.dump(metricas, archivo, indent=2)

            mlflow.log_artifacts(str(carpeta), artifact_path="p2")

    print("\nBaseline P2 registrado en MLflow.")
