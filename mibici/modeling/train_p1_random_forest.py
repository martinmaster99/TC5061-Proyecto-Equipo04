import subprocess

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

RUTA = "data/processed/p1_features_final.parquet"
COLUMNAS = [
    "origen_id",
    "hora",
    "dia_semana",
    "mes",
    "fin_semana",
    "viajes",
    "viajes_hace_1h",
    "viajes_hace_24h",
    "promedio_24h",
]
OBJETIVO = "target_siguiente_hora"

print("[1/4] Cargando datos...")
df = pd.read_parquet(RUTA)
df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])

train = df[df["fecha_hora"] < "2026-05-01"]
val = df[
    (df["fecha_hora"] >= "2026-05-01")
    & (df["fecha_hora"] < "2026-06-01")
]
test = df[df["fecha_hora"] >= "2026-06-01"]

X_train = train[COLUMNAS]
y_train = train[OBJETIVO]
X_val = val[COLUMNAS]
y_val = val[OBJETIVO]
X_test = test[COLUMNAS]
y_test = test[OBJETIVO]

print("[2/4] Entrenando Random Forest...")
modelo = RandomForestRegressor(
    n_estimators=50,
    max_depth=12,
    min_samples_leaf=10,
    max_features=0.8,
    random_state=42,
    n_jobs=2,
)
modelo.fit(X_train, y_train)

print("[3/4] Calculando métricas...")

def evaluar(X, y):
    pred = np.maximum(modelo.predict(X), 0)
    return {
        "mae": mean_absolute_error(y, pred),
        "rmse": np.sqrt(mean_squared_error(y, pred)),
    }

resultados = {
    "train": evaluar(X_train, y_train),
    "val": evaluar(X_val, y_val),
    "test": evaluar(X_test, y_test),
}

for conjunto, metricas in resultados.items():
    print(
        f"{conjunto.upper()}: "
        f"MAE={metricas['mae']:.4f} | "
        f"RMSE={metricas['rmse']:.4f}"
    )

print("[4/4] Registrando en MLflow...")

commit = subprocess.check_output(
    ["git", "rev-parse", "--short", "HEAD"],
    text=True,
).strip()

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("mibici-equipo04")

with mlflow.start_run(run_name="P1_random_forest"):
    mlflow.log_params({
        "modelo": "RandomForestRegressor",
        "n_estimators": 50,
        "max_depth": 12,
        "min_samples_leaf": 10,
        "max_features": 0.8,
        "semilla": 42,
        "fecha_inicio": "2026-01-01",
        "fecha_fin": "2026-06-30",
        "datos_version": "mibici_2026_01_06",
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
    })

    for conjunto, metricas in resultados.items():
        for nombre, valor in metricas.items():
            mlflow.log_metric(f"{conjunto}_{nombre}", float(valor))

    mlflow.set_tags({
        "equipo": "04",
        "pregunta": "P1",
        "git_commit": commit,
        "modelo": "random_forest",
    })

print("Random Forest registrado correctamente en MLflow.")
