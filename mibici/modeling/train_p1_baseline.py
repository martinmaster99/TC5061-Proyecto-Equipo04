"""Baseline P1: predicción de demanda mediante persistencia."""

import subprocess

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from mibici.config import PROCESSED_DATA_DIR, EXPERIMENT_NAME


def evaluar(y_real, y_pred):
    return {
        "mae": mean_absolute_error(y_real, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_real, y_pred)),
    }


def main():
    ruta = PROCESSED_DATA_DIR / "p1_features_final.parquet"
    df = pd.read_parquet(ruta)

    # Evitamos que las etiquetas crucen los límites temporales.
    train = df[df["fecha_hora"] < "2026-04-30 23:00:00"]
    val = df[
        (df["fecha_hora"] >= "2026-05-01") &
        (df["fecha_hora"] < "2026-05-31 23:00:00")
    ]
    test = df[df["fecha_hora"] >= "2026-06-01"]

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
    )
    git_commit = (
        commit.stdout.strip()
        if commit.returncode == 0
        else "sin_commit"
    )

    with mlflow.start_run(run_name="P1_baseline_persistencia"):
        mlflow.set_tags({
            "equipo": "04",
            "pregunta": "P1",
            "git_commit": git_commit,
            "modelo": "persistencia",
        })

        mlflow.log_params({
            "fecha_inicio": "2026-01-01",
            "fecha_fin": "2026-06-30",
            "version_datos": "mibici_2026_01_06",
            "horizonte_horas": 1,
            "seed": 42,
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
        })

        for nombre, datos in [
            ("train", train),
            ("val", val),
            ("test", test),
        ]:
            y_real = datos["target_siguiente_hora"]
            y_pred = datos["viajes"]

            metricas = evaluar(y_real, y_pred)

            for metrica, valor in metricas.items():
                mlflow.log_metric(
                    f"{nombre}_{metrica}",
                    float(valor),
                )

            print(
                f"{nombre.upper()}: "
                f"MAE={metricas['mae']:.4f} | "
                f"RMSE={metricas['rmse']:.4f}"
            )

        print("\nBaseline registrado correctamente en MLflow.")


if __name__ == "__main__":
    main()
