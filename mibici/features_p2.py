from pathlib import Path
import pandas as pd

CARPETA = Path("data/processed")
ARCHIVOS = sorted(CARPETA.glob("datos_abiertos_2026_*_integrado.parquet"))

if len(ARCHIVOS) != 6:
    raise RuntimeError(f"Se esperaban 6 archivos, encontrados: {len(ARCHIVOS)}")

movimientos = []

for archivo in ARCHIVOS:
    df = pd.read_parquet(
        archivo,
        columns=[
            "inicio_del_viaje",
            "fin_del_viaje",
            "origen_id",
            "destino_id",
        ],
    )

    salidas = pd.DataFrame({
        "estacion_id": df["origen_id"],
        "fecha_hora": df["inicio_del_viaje"].dt.floor("h"),
        "salidas": 1,
        "entradas": 0,
    })

    entradas = pd.DataFrame({
        "estacion_id": df["destino_id"],
        "fecha_hora": df["fin_del_viaje"].dt.floor("h"),
        "salidas": 0,
        "entradas": 1,
    })

    movimientos.append(salidas)
    movimientos.append(entradas)

    print(f"[OK] {archivo.name}: {len(df):,} viajes")

movimientos = pd.concat(movimientos, ignore_index=True)

flujos = (
    movimientos
    .dropna(subset=["estacion_id", "fecha_hora"])
    .groupby(["estacion_id", "fecha_hora"], as_index=False)
    [["salidas", "entradas"]]
    .sum()
)

flujos["flujo_neto"] = flujos["entradas"] - flujos["salidas"]

destino = CARPETA / "p2_flujos_horarios.parquet"
flujos.to_parquet(destino, index=False)

print("\n===== RESULTADOS P2 =====")
print("Registros estación-hora:", len(flujos))
print("Estaciones:", flujos["estacion_id"].nunique())
print("Total salidas:", flujos["salidas"].sum())
print("Total entradas:", flujos["entradas"].sum())
print("Archivo generado:", destino)
