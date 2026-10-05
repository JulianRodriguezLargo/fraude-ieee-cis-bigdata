"""Capa SILVER: datos limpios y validados, listos para EDA y para construir features.

Aplica el plan de Veracity de la Fase 1:
  1. Normaliza nombres de columnas (test_identity usa "id-01" y train_identity "id_01").
  2. Elimina duplicados de TransactionID.
  3. Une transaction + identity con LEFT JOIN (solo ~24 % tiene identidad).
  4. Une train y test en una sola tabla con la columna `dataset`.
  5. Elimina las columnas con más de 90 % de nulos, calculado SOLO con el train
     (para no usar información del test) y guarda la lista en _meta/.
  6. Deriva día, hora y día de la semana desde TransactionDT (viene en segundos).
  7. Normaliza categóricas: email -> proveedor, DeviceInfo -> marca,
     categorías casi vacías de card6 -> "other".

Los nulos restantes NO se imputan aquí: el patrón de nulos es información
(donde hay identidad el fraude sube a 6,5-10 %). La imputación y los flags
is_null_* se hacen en Gold, al construir las features del modelo.

Uso:
    python -m src.silver_limpieza
"""
import json
import time

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.config import BRONZE, META, SILVER, TABLE_FORMAT, UMBRAL_NULOS, get_logger, get_spark, ruta

log = get_logger("silver")

AUDITORIA = ["_ingest_ts", "_source_file"]
# Columnas que nunca se eliminan aunque tengan nulos
PROTEGIDAS = {"TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD", "dataset"}
SEGUNDOS_DIA = 86_400


# ---------------------------------------------------------------------------
# Pasos de limpieza (funciones puras sobre DataFrames: fáciles de probar)
# ---------------------------------------------------------------------------
def normalizar_nombres(df: DataFrame) -> DataFrame:
    """id-01 -> id_01. En el test de Kaggle las columnas de identidad usan guion."""
    return df.toDF(*[c.replace("-", "_") for c in df.columns])


def leer_bronze(spark: SparkSession, tabla: str) -> DataFrame:
    df = spark.read.format(TABLE_FORMAT).load(ruta(BRONZE, tabla))
    return normalizar_nombres(df.drop(*AUDITORIA)).dropDuplicates(["TransactionID"])


def unir_transaccion_identidad(tx: DataFrame, idn: DataFrame, dataset: str) -> DataFrame:
    """LEFT JOIN: conserva todas las transacciones, tengan o no identidad.
    Agrega `tiene_identidad` (True si la transacción aparece en la tabla de identidad)."""
    idn = idn.withColumn("tiene_identidad", F.lit(True))
    return (
        tx.join(idn, on="TransactionID", how="left")
        .withColumn("tiene_identidad", F.coalesce(F.col("tiene_identidad"), F.lit(False)))
        .withColumn("dataset", F.lit(dataset))
    )


def fraccion_nulos(df: DataFrame) -> dict:
    """Fracción de nulos de cada columna, calculada en una sola pasada sobre los datos."""
    fila = df.select(
        [F.avg(F.col(c).isNull().cast("double")).alias(c) for c in df.columns]
    ).first()
    return {c: float(fila[c]) for c in df.columns}


def columnas_a_eliminar(nulos_train: dict) -> list:
    return sorted(c for c, f in nulos_train.items() if f > UMBRAL_NULOS and c not in PROTEGIDAS)


def derivar_tiempo(df: DataFrame) -> DataFrame:
    """TransactionDT son segundos desde una fecha de referencia oculta.
    Día, hora y día de la semana son RELATIVOS (no son la fecha real)."""
    return (
        df.withColumn("dia", F.floor(F.col("TransactionDT") / SEGUNDOS_DIA).cast("int"))
        .withColumn("hora", (F.floor(F.col("TransactionDT") / 3600) % 24).cast("int"))
        .withColumn("dia_semana", (F.col("dia") % 7).cast("int"))
    )


def proveedor_email(col: str) -> F.Column:
    """gmail.com -> gmail · yahoo.com.mx -> yahoo · hotmail.co.uk -> hotmail"""
    limpio = F.lower(F.trim(F.col(col)))
    return F.when(limpio.isNull() | (limpio == ""), None).otherwise(F.split(limpio, r"\.").getItem(0))


def marca_dispositivo(col: str = "DeviceInfo") -> F.Column:
    """Agrupa las cientos de variantes de DeviceInfo en marcas."""
    d = F.lower(F.trim(F.col(col)))
    return (
        F.when(d.isNull() | (d == ""), None)
        .when(d.startswith("sm-") | d.contains("samsung") | d.startswith("gt-"), "samsung")
        .when(d.contains("ios device") | d.contains("macos") | d.contains("iphone"), "apple")
        .when(d.startswith("moto") | d.contains("motorola") | d.startswith("xt"), "motorola")
        .when(d.startswith("lg") | d.contains("lg-"), "lg")
        .when(d.contains("huawei") | d.startswith("ale-") | d.startswith("ane-"), "huawei")
        .when(d.contains("redmi") | d.contains("xiaomi") | d.startswith("mi "), "xiaomi")
        .when(d.contains("pixel") | d.contains("nexus"), "google")
        .when(d.contains("windows") | d.contains("trident"), "windows")
        .when(d.contains("linux"), "linux")
        .otherwise("otro")
    )


def normalizar_categoricas(df: DataFrame) -> DataFrame:
    # Recortar espacios en todas las columnas de texto
    for c, tipo in df.dtypes:
        if tipo == "string":
            df = df.withColumn(c, F.trim(F.col(c)))
    if "P_emaildomain" in df.columns:
        df = df.withColumn("P_email_proveedor", proveedor_email("P_emaildomain"))
    if "R_emaildomain" in df.columns:
        df = df.withColumn("R_email_proveedor", proveedor_email("R_emaildomain"))
    if "DeviceInfo" in df.columns:
        df = df.withColumn("device_marca", marca_dispositivo("DeviceInfo"))
    if "card6" in df.columns:
        # "charge card" (15 filas) y "debit or credit" (30 filas): demasiado raras para aprender algo
        df = df.withColumn(
            "card6",
            F.when(F.col("card6").isin("credit", "debit") | F.col("card6").isNull(), F.col("card6"))
            .otherwise(F.lit("other")),
        )
    return df


# ---------------------------------------------------------------------------
# Orquestación de la capa
# ---------------------------------------------------------------------------
def run(spark: SparkSession) -> dict:
    inicio = time.time()
    log.info("=== SILVER: limpieza desde Bronze ===")

    partes = []
    for dataset in ("train", "test"):
        tx = leer_bronze(spark, f"{dataset}_transaction")
        idn = leer_bronze(spark, f"{dataset}_identity")
        partes.append(unir_transaccion_identidad(tx, idn, dataset))
        log.info("%-5s: transaction ⋈ identity (left join) listo", dataset)

    # train y test en una sola tabla; el test no trae isFraud (queda nulo)
    df = partes[0].unionByName(partes[1], allowMissingColumns=True)

    # Columnas a eliminar: se decide SOLO con el train
    nulos = fraccion_nulos(df.filter(F.col("dataset") == "train"))
    eliminar = columnas_a_eliminar(nulos)
    log.info("Columnas con más de %.0f %% de nulos en el train: %d -> se eliminan",
             UMBRAL_NULOS * 100, len(eliminar))
    df = df.drop(*eliminar)

    df = derivar_tiempo(df)
    df = normalizar_categoricas(df)

    destino = ruta(SILVER, "transacciones")
    (
        df.write.format(TABLE_FORMAT)
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .partitionBy("dataset")
        .save(destino)
    )

    # Resumen guardado como evidencia de que el pipeline corrió
    silver = spark.read.format(TABLE_FORMAT).load(destino)
    conteo = {r["dataset"]: r["count"] for r in silver.groupBy("dataset").count().collect()}
    train = silver.filter(F.col("dataset") == "train")
    fraudes = train.filter(F.col("isFraud") == 1).count()
    con_identidad = train.filter(F.col("tiene_identidad")).count()
    resumen = {
        "filas_por_dataset": conteo,
        "columnas_silver": len(silver.columns),
        "columnas_eliminadas_por_nulos": len(eliminar),
        "umbral_nulos": UMBRAL_NULOS,
        "tasa_fraude_train_pct": round(100 * fraudes / max(conteo.get("train", 1), 1), 2),
        "pct_train_con_identidad": round(100 * con_identidad / max(conteo.get("train", 1), 1), 1),
        "segundos": round(time.time() - inicio, 1),
    }
    META.mkdir(parents=True, exist_ok=True)
    (META / "silver_columnas_eliminadas.json").write_text(
        json.dumps({c: round(nulos[c], 4) for c in eliminar}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (META / "silver_resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")

    log.info("Silver escrito en %s", destino)
    for k, v in resumen.items():
        log.info("  %-32s %s", k, v)
    return resumen


if __name__ == "__main__":
    spark = get_spark("silver-limpieza")
    run(spark)
    spark.stop()
