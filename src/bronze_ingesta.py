"""Capa BRONZE: carga los CSV crudos al lakehouse sin transformarlos (ELT).

Reglas de Bronze (diseño de la Fase 1):
  * Los datos se guardan tal como llegan: mismas columnas y mismos nombres.
  * Solo se agregan dos columnas de auditoría: _ingest_ts (cuándo se cargó)
    y _source_file (de qué archivo vino).
  * Es append-only: nunca se modifica ni se borra. Si un archivo ya fue
    cargado, se omite, para que volver a correr el pipeline no duplique datos.

Uso:
    python -m src.bronze_ingesta
"""
import time

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.config import BRONZE, DATA_RAW, FUENTES_IEEE, TABLE_FORMAT, existe_tabla, get_logger, get_spark, ruta

log = get_logger("bronze")


def leer_csv(spark: SparkSession, archivo: str) -> DataFrame:
    """Lee un CSV de IEEE-CIS con encabezado e inferencia de tipos."""
    return (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(str(DATA_RAW / archivo))
    )


def ya_cargado(spark: SparkSession, destino: str, archivo: str) -> bool:
    """True si ese archivo ya está en la tabla Bronze (evita duplicar al re-ejecutar)."""
    if not existe_tabla(spark, destino):
        return False
    cargados = (
        spark.read.format(TABLE_FORMAT).load(destino)
        .select("_source_file").distinct().collect()
    )
    return any(fila["_source_file"].endswith(archivo) for fila in cargados)


def ingestar_archivo(spark: SparkSession, archivo: str, tabla: str) -> int:
    """Carga un CSV a su tabla Bronze. Devuelve el número de filas cargadas (0 si se omitió)."""
    origen = DATA_RAW / archivo
    destino = ruta(BRONZE, tabla)

    if not origen.exists():
        log.warning("No existe %s: se omite. ¿Descargaste los datos en data/raw?", origen)
        return 0
    if ya_cargado(spark, destino, archivo):
        log.info("%s ya estaba en Bronze: se omite (append-only, sin duplicados)", archivo)
        return 0

    inicio = time.time()
    df = (
        leer_csv(spark, archivo)
        .withColumn("_ingest_ts", F.current_timestamp())
        .withColumn("_source_file", F.col("_metadata.file_path"))
    )
    df.write.format(TABLE_FORMAT).mode("append").save(destino)

    filas = spark.read.format(TABLE_FORMAT).load(destino).filter(
        F.col("_source_file").endswith(archivo)
    ).count()
    log.info("%-24s -> bronze/%-18s %10s filas  %4d columnas  (%.0f s)",
             archivo, tabla, f"{filas:,}", len(df.columns) - 2, time.time() - inicio)
    return filas


def run(spark: SparkSession) -> dict:
    """Ingesta todas las fuentes de IEEE-CIS. Devuelve {tabla: filas cargadas}."""
    log.info("=== BRONZE: ingesta de %d archivos desde %s ===", len(FUENTES_IEEE), DATA_RAW)
    return {tabla: ingestar_archivo(spark, archivo, tabla) for archivo, tabla in FUENTES_IEEE.items()}


if __name__ == "__main__":
    spark = get_spark("bronze-ingesta")
    run(spark)
    spark.stop()
