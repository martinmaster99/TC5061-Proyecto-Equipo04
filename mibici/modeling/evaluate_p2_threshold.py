import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

df = pd.read_parquet(
    "data/processed/p2_features_final.parquet"
)

df["fecha_objetivo"] = (
    df["fecha_hora"] + pd.Timedelta(hours=1)
)

train = df[df["fecha_objetivo"] < "2026-05-01"]
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

modelo = DecisionTreeClassifier(
    max_depth=8,
    min_samples_leaf=100,
    class_weight="balanced",
    random_state=42,
)

modelo.fit(
    train[variables],
    train["target_siguiente_hora"]
)

umbral = 0.90

probabilidades = modelo.predict_proba(test[variables])
clases = modelo.classes_

prob_riesgos = probabilidades[:, clases != 0]
clases_riesgo = clases[clases != 0]

mejor_indice = prob_riesgos.argmax(axis=1)
mejor_probabilidad = prob_riesgos.max(axis=1)

pred = np.zeros(len(test), dtype=int)
activar = mejor_probabilidad >= umbral
pred[activar] = clases_riesgo[mejor_indice[activar]]

real = test["target_siguiente_hora"]

metricas = {
    "accuracy": accuracy_score(real, pred),
    "balanced_accuracy": balanced_accuracy_score(real, pred),
    "f1_macro": f1_score(
        real, pred, labels=[0, 1, 2],
        average="macro", zero_division=0
    ),
    "precision_vaciado": precision_score(
        real, pred, labels=[1],
        average="macro", zero_division=0
    ),
    "precision_saturacion": precision_score(
        real, pred, labels=[2],
        average="macro", zero_division=0
    ),
    "recall_vaciado": recall_score(
        real, pred, labels=[1],
        average="macro", zero_division=0
    ),
    "recall_saturacion": recall_score(
        real, pred, labels=[2],
        average="macro", zero_division=0
    ),
    "alertas_generadas": int((pred != 0).sum()),
}

carpeta = Path("reports/p2/arbol_umbral_090")
carpeta.mkdir(parents=True, exist_ok=True)

matriz = confusion_matrix(real, pred, labels=[0, 1, 2])

pd.DataFrame(
    matriz,
    index=["Real_0", "Real_1", "Real_2"],
    columns=["Pred_0", "Pred_1", "Pred_2"],
).to_csv(carpeta / "matriz_confusion.csv")

with open(
    carpeta / "metricas_test.json",
    "w",
    encoding="utf-8"
) as archivo:
    json.dump(metricas, archivo, indent=2)

mlflow.set_experiment("mibici-equipo04")

with mlflow.start_run(
    run_name="P2_arbol_umbral_090"
):
    mlflow.log_params({
        "pregunta": "P2",
        "modelo": "DecisionTreeClassifier",
        "max_depth": 8,
        "min_samples_leaf": 100,
        "class_weight": "balanced",
        "umbral": umbral,
        "umbral_seleccionado_con": "validacion_mayo",
        "version_features": "p2_features_final_v1",
        "train_periodo": "2026-01_a_2026-04",
        "val_periodo": "2026-05",
        "test_periodo": "2026-06",
    })

    for nombre, valor in metricas.items():
        mlflow.log_metric(f"test_{nombre}", valor)

    mlflow.log_artifacts(
        str(carpeta),
        artifact_path="p2_umbral_090"
    )

    print("===== TEST JUNIO - UMBRAL 0.90 =====")

    for nombre, valor in metricas.items():
        print(f"{nombre}: {valor:.4f}")

    print("\n===== REPORTE DE CLASIFICACIÓN =====")
    print(
        classification_report(
            real, pred,
            labels=[0, 1, 2],
            zero_division=0
        )
    )

    print("\n===== MATRIZ DE CONFUSIÓN =====")
    print(matriz)

    print("\nEvaluación P2 registrada en MLflow.")
