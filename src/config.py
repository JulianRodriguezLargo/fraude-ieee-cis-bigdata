"""Configuración común del pipeline: rutas del lakehouse y creación de la sesión de Spark.

Todo se controla con variables de entorno (archivo .env), para que el mismo código
funcione en Docker, en el equipo de cada integrante o en Colab sin cambiar nada.
"""
import logging
import os
from pathlib import Path

from pyspark.sql import SparkSession

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent.parent
DATA_RAW = Path(os.getenv("DATA_RAW", RAIZ / "data" / "raw"))
LAKEHOUSE = Path(os.getenv("LAKEHOUSE", RAIZ / "data" / "lakehouse"))
if not DATA_RAW.is_absolute():
    DATA_RAW = RAIZ / DATA_RAW
if not LAKEHOUSE.is_absolute():
    LAKEHOUSE = RAIZ / LAKEHOUSE

BRONZE = LAKEHOUSE / "bronze"
SILVER = LAKEHOUSE / "silver"
GOLD = LAKEHOUSE / "gold"
META = LAKEHOUSE / "_meta"  # decisiones del pipeline guardadas para reproducibilidad

# delta (lo normal) o parquet (solo para pruebas en un entorno sin los JAR de Delta)
TABLE_FORMAT = os.getenv("TABLE_FORMAT", "delta").lower()

# Archivos de origen de IEEE-CIS -> nombre de la tabla Bronze
FUENTES_IEEE = {
    "train_transaction.csv": "train_transaction",
    "train_identity.csv": "train_identity",
    "test_transaction.csv": "test_transaction",
    "test_identity.csv": "test_identity",
}

# Umbral de la Fase 1: se eliminan las columnas con más de 90 % de nulos en el train
UMBRAL_NULOS = 0.90


def ruta(capa: Path, tabla: str) -> str:
    """Ruta de una tabla del lakehouse como texto (Spark la necesita así)."""
    return str(capa / tabla)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def get_logger(nombre: str) -> logging.Logger:
    """Logger que escribe en pantalla y en data/lakehouse/_meta/pipeline.log (evidencia de ejecución)."""
    raiz = logging.getLogger()
    if not raiz.handlers:
        formato = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")
        consola = logging.StreamHandler()
        consola.setFormatter(formato)
        raiz.addHandler(consola)
        META.mkdir(parents=True, exist_ok=True)
        archivo = logging.FileHandler(META / "pipeline.log", encoding="utf-8")
        archivo.setFormatter(formato)
        raiz.addHandler(archivo)
        raiz.setLevel(logging.INFO)
        logging.getLogger("py4j").setLevel(logging.WARNING)
    return logging.getLogger(nombre)


# ---------------------------------------------------------------------------
# Spark
# ---------------------------------------------------------------------------
def get_spark(app_name: str = "fraude-ieee-cis", kafka: bool = False) -> SparkSession:
    """Crea (o reutiliza) la sesión de Spark, con Delta Lake si TABLE_FORMAT=delta.

    Con kafka=True agrega el conector de Kafka para Spark (lo usa el streaming, que es opcional).
    """
    paquetes = []
    if kafka:
        import pyspark

        paquetes.append(f"org.apache.spark:spark-sql-kafka-0-10_2.12:{pyspark.__version__}")
    builder = (
        SparkSession.builder.appName(app_name)
        .master(os.getenv("SPARK_MASTER", "local[*]"))
        .config("spark.driver.memory", os.getenv("SPARK_DRIVER_MEMORY", "4g"))
        # 434 columnas: particiones moderadas para no generar miles de archivos pequeños
        .config("spark.sql.shuffle.partitions", os.getenv("SPARK_SHUFFLE_PARTITIONS", "16"))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.showConsoleProgress", "false")
    )
    if TABLE_FORMAT == "delta":
        from delta import configure_spark_with_delta_pip

        builder = (
            builder.config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        )
        spark = configure_spark_with_delta_pip(builder, extra_packages=paquetes).getOrCreate()
    else:
        if paquetes:
            builder = builder.config("spark.jars.packages", ",".join(paquetes))
        spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def existe_tabla(spark: SparkSession, path: str) -> bool:
    """True si ya hay una tabla escrita en esa ruta."""
    if TABLE_FORMAT == "delta":
        from delta.tables import DeltaTable

        return DeltaTable.isDeltaTable(spark, path)
    return (Path(path) / "_SUCCESS").exists()
