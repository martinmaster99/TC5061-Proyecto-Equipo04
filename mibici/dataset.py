"""
Proyecto MiBici - Equipo04
Descarga reproducible de datos abiertos.
Periodo: enero a junio de 2026.
"""

from pathlib import Path
import csv
import requests
import typer

from mibici.config import RAW_DATA_DIR

app = typer.Typer()

# Enlaces oficiales de MiBici
URLS = {
    "2026_01": "https://mibici.net/site/assets/files/120163/datos_abiertos_2026_01.csv",
    "2026_02": "https://mibici.net/site/assets/files/121761/datos_abiertos_2026_02.csv",
    "2026_03": "https://mibici.net/site/assets/files/123236/datos_abiertos_2026_03.csv",
    "2026_04": "https://mibici.net/site/assets/files/124927/datos_abiertos_2026_04.csv",
    "2026_05": "https://mibici.net/site/assets/files/125943/datos_abiertos_2026_05.csv",
    "2026_06": "https://mibici.net/site/assets/files/127734/datos_abiertos_2026_06.csv",
}


def descargar_archivo(url: str, destino: Path):
    """Descarga y valida un CSV utilizando curl."""

    import subprocess

    destino.parent.mkdir(parents=True, exist_ok=True)

    if destino.exists() and destino.stat().st_size > 0:
        print(f"[EXISTE] {destino.name}")
        return

    print(f"[DESCARGANDO] {destino.name}")

    temporal = destino.with_suffix(".csv.part")

    try:
        subprocess.run(
            [
                "curl",
                "-L",
                "--fail",
                "--retry", "2",
                "--max-time", "180",
                "--silent",
                "--show-error",
                url,
                "-o", str(temporal),
            ],
            check=True,
        )

        if not temporal.exists() or temporal.stat().st_size == 0:
            raise ValueError("Archivo vacío")

        with temporal.open(
            "r",
            encoding="utf-8-sig",
            errors="replace",
            newline=""
        ) as archivo:
            encabezados = next(csv.reader(archivo))

        if len(encabezados) < 5:
            raise ValueError("CSV inválido")

        temporal.replace(destino)

        print(
            f"[OK] {destino.name} "
            f"({destino.stat().st_size / 1024**2:.2f} MB)"
        )

    except Exception:
        temporal.unlink(missing_ok=True)
        raise


@app.command()
def main():
    """Descarga los seis meses de viajes de MiBici."""

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for mes, url in URLS.items():

        nombre = f"datos_abiertos_{mes}.csv"
        destino = RAW_DATA_DIR / nombre

        try:
            descargar_archivo(url, destino)

        except Exception as error:
            print(f"[ERROR] {nombre}: {error}")
            raise typer.Exit(code=1)

    print("\nDescarga de viajes finalizada.")


if __name__ == "__main__":
    app()
