import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
)

df = pd.read_parquet(
    "data/processed/p2_features_final.parquet"
)

df["fecha_objetivo"] = (
    df["fecha_hora"] + pd.Timedelta(hours=1)
)

train = df[df["fecha_objetivo"] < "2026-05-01"]
val = df[
    (df["fecha_objetivo"] >= "2026-05-01") &
    (df["fecha_objetivo"] < "2026-06-01")
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

probabilidades = modelo.predict_proba(val[variables])
clases = modelo.classes_

print("===== UMBRALES EN VALIDACIÓN (MAYO) =====")

resultados = []

for umbral in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]:
    pred = np.zeros(len(val), dtype=int)

    prob_riesgos = probabilidades[:, clases != 0]
    clases_riesgo = clases[clases != 0]

    mejor_indice = prob_riesgos.argmax(axis=1)
    mejor_probabilidad = prob_riesgos.max(axis=1)

    activar = mejor_probabilidad >= umbral
    pred[activar] = clases_riesgo[mejor_indice[activar]]

    resultados.append({
        "umbral": umbral,
        "f1_macro": f1_score(
            val["target_siguiente_hora"],
            pred,
            average="macro",
            labels=[0, 1, 2],
            zero_division=0,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            val["target_siguiente_hora"], pred
        ),
        "precision_vaciado": precision_score(
            val["target_siguiente_hora"], pred,
            labels=[1], average="macro", zero_division=0
        ),
        "precision_saturacion": precision_score(
            val["target_siguiente_hora"], pred,
            labels=[2], average="macro", zero_division=0
        ),
        "recall_vaciado": recall_score(
            val["target_siguiente_hora"], pred,
            labels=[1], average="macro", zero_division=0
        ),
        "recall_saturacion": recall_score(
            val["target_siguiente_hora"], pred,
            labels=[2], average="macro", zero_division=0
        ),
        "alertas": int((pred != 0).sum()),
    })

tabla = pd.DataFrame(resultados)

print(tabla.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

mejor = tabla.loc[tabla["f1_macro"].idxmax()]

print("\n===== MEJOR UMBRAL SEGÚN F1 MACRO =====")
print(mejor.to_string())
