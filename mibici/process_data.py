"""Limpieza y conversión de viajes MiBici a Parquet."""

import hashlib
import json
import unicodedata
from pathlib import Path

import pandas as pd

from mibici.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MANIFEST_PATH,
)


def normalizar_columna(nombre):
    """Normaliza nombres de columnas y elimina acentos."""
    nombre = unicodedata.normalize("NFKD", nombre)
    nombre = "".join(
        c for c in nombre if not unicodedata.combining(c)
    )
    return nombre.strip().lower().replace(" ", "_")


def calcular_hash(ruta):
    """Calcula SHA-256 sin cargar todo el archivo en memoria."""
    sha = hashlib.sha256()

    with open(ruta, "rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            sha.update(bloque)

    return sha.hexdigest()


def procesar_archivo(ruta):
    """Limpia viajes y conserva variables operativas."""

    print(f"\n[PROCESANDO] {ruta.name}")

    df = pd.read_csv(
        ruta,
        dtype=str,
        encoding="utf-8-sig",
        encoding_errors="replace"
    )

    filas_originales = len(df)

    df.columns = [
        normalizar_columna(c) for c in df.columns
    ]

    columnas = [
        "viaje_id",
        "inicio_del_viaje",
        "fin_del_viaje",
        "origen_id",
        "destino_id",
    ]

    faltantes = set(columnas) - set(df.columns)

    if faltantes:
        raise ValueError(f"Columnas faltantes: {faltantes}")

    df = df[columnas].copy()

    df["inicio_del_viaje"] = pd.to_datetime(
        df["inicio_del_viaje"], errors="coerce"
    )

    df["fin_del_viaje"] = pd.to_datetime(
        df["fin_del_viaje"], errors="coerce"
    )

    df = df.dropna(subset=columnas)
    df = df.drop_duplicates(subset=["viaje_id"])

    df["duracion_minutos"] = (
        df["fin_del_viaje"] - df["inicio_del_viaje"]
    ).dt.total_seconds() / 60

    df = df[df["duracion_minutos"] > 0].copy()

    df["hora"] = df["inicio_del_viaje"].dt.hour
    df["dia_semana"] = df["inicio_del_viaje"].dt.dayofweek
    df["mes"] = df["inicio_del_viaje"].dt.month

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    destino = PROCESSED_DATA_DIR / (
        ruta.stem + ".parquet"
    )

    df.to_parquet(destino, index=False, engine="pyarrow")

    print(f"[OK] {len(df):,} viajes procesados")

    return {
        "archivo_original": ruta.name,
        "archivo_procesado": destino.name,
        "filas_originales": filas_originales,
        "filas_procesadas": len(df),
        "sha256_original": calcular_hash(ruta),
        "sha256_procesado": calcular_hash(destino),
    }


def main():
    archivos = sorted(
        RAW_DATA_DIR.glob("datos_abiertos_2026_*.csv")
    )

    if len(archivos) != 6:
        raise RuntimeError(
            f"Se esperaban 6 CSV y se encontraron {len(archivos)}"
        )

    resultados = [
        procesar_archivo(archivo) for archivo in archivos
    ]

    manifiesto = {
        "proyecto": "MiBici",
        "equipo": "04",
        "periodo": "2026-01 a 2026-06",
        "algoritmo_hash": "SHA-256",
        "archivos": resultados,
    }

    MANIFEST_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    MANIFEST_PATH.write_text(
        json.dumps(
            manifiesto,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\nProcesamiento finalizado.")
    print("Manifiesto:", MANIFEST_PATH)


if __name__ == "__main__":
    main()
