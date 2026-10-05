"""Ejecuta el pipeline completo en orden, de punta a punta.

Es el orquestador de la etapa 1. Cuando se agregue Airflow (etapa 3), el DAG
llamará exactamente a estas mismas funciones, así que no se duplica lógica.

Uso:
    python -m src.pipeline                                   # todo: bronze silver gold_kpis gold_features modelo
    python -m src.pipeline --etapas bronze silver            # solo algunas etapas
    python -m src.pipeline --etapas modelo --fraccion 0.2    # modelo rápido con 20 % de las legítimas
"""
import argparse
import shutil
import time

from src import bronze_ingesta, gold_features, gold_kpis, modelo, silver_limpieza
from src.config import META, RAIZ, get_logger, get_spark

log = get_logger("pipeline")

# Orden del pipeline Medallion
ETAPAS = ["bronze", "silver", "gold_kpis", "gold_features", "modelo"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline Medallion de detección de fraude (IEEE-CIS)")
    parser.add_argument("--etapas", nargs="+", choices=ETAPAS, default=ETAPAS,
                        help="Etapas a ejecutar, en orden (por defecto: todas)")
    parser.add_argument("--fraccion", type=float, default=1.0,
                        help="Modelo: fracción de transacciones legítimas para entrenar (los fraudes se usan todos)")
    parser.add_argument("--iteraciones", type=int, default=60, help="Modelo: número de árboles del GBT")
    args = parser.parse_args()

    funciones = {
        "bronze": bronze_ingesta.run,
        "silver": silver_limpieza.run,
        "gold_kpis": gold_kpis.run,
        "gold_features": gold_features.run,
        "modelo": lambda spark: modelo.run(spark, args.fraccion, args.iteraciones),
    }

    spark = get_spark("pipeline-fraude")
    inicio = time.time()
    try:
        for etapa in [e for e in ETAPAS if e in args.etapas]:  # siempre en el orden correcto
            t = time.time()
            funciones[etapa](spark)
            log.info(">>> Etapa %s terminada en %.0f s", etapa, time.time() - t)
        log.info(">>> Pipeline completo en %.0f s", time.time() - inicio)
    except Exception:
        log.exception(">>> El pipeline falló después de %.0f s", time.time() - inicio)
        raise
    finally:
        copiar_evidencias()  # también si falla: el log es la evidencia del error
        spark.stop()


def copiar_evidencias() -> None:
    """Copia el log y los resúmenes JSON a docs/evidencias/ (data/ no se sube a Git)."""
    destino = RAIZ / "docs" / "evidencias"
    destino.mkdir(parents=True, exist_ok=True)
    for archivo in list(META.glob("*.json")) + [META / "pipeline.log"]:
        if archivo.exists():
            shutil.copy2(archivo, destino / archivo.name)
    log.info("Evidencias (log y resúmenes JSON) copiadas a %s", destino)


if __name__ == "__main__":
    main()
