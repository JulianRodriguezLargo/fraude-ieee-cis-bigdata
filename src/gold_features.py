"""Capa GOLD (features): tabla lista para entrenar el modelo de fraude.

Decisiones (plan de Veracity de la Fase 1):
  * Numéricas: los nulos se reemplazan por -999 (valor centinela). Los árboles
    aprenden que -999 significa "faltaba", así que no se pierde la información del nulo.
  * `n_nulos`: cuántos campos venían vacíos en la transacción. El patrón de nulos es señal.
  * `log_monto`: log(1 + TransactionAmt), para que los montos gigantes (outliers) no dominen.
  * Categóricas: nulo -> "missing". Las de más de 60 valores distintos se excluyen
    (para eso existen las versiones agrupadas: P_email_proveedor, device_marca).
  * División por TIEMPO, no al azar: los últimos 30 días del train son la validación.
    Así se evalúa como en la vida real: entrenar con el pasado y predecir el futuro.

Columnas de la tabla gold/features:
    TransactionID, isFraud, TransactionAmt, split (entrenamiento | validacion | test_kaggle),
    + features numéricas y categóricas.
La lista de features usadas queda en _meta/gold_features.json.

Uso:
    python -m src.gold_features
"""
import json

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.config import GOLD, META, SILVER, TABLE_FORMAT, get_logger, get_spark, ruta

log = get_logger("gold_features")

CENTINELA = -999
MAX_CATEGORIAS = 60
DIAS_VALIDACION = 30

# No son features: identificadores, etiqueta, tiempo absoluto o columnas reemplazadas por versiones agrupadas
EXCLUIR = {
    "TransactionID", "isFraud", "TransactionDT", "dia", "dataset",
    "P_emaildomain", "R_emaildomain", "DeviceInfo",
}


def clasificar_columnas(df: DataFrame) -> tuple:
    numericas, categoricas = [], []
    for c, tipo in df.dtypes:
        if c in EXCLUIR:
            continue
        (categoricas if tipo == "string" else numericas).append(c)
    return numericas, categoricas


def filtrar_alta_cardinalidad(train: DataFrame, categoricas: list) -> tuple:
    """Separa las categóricas manejables de las que tienen demasiados valores distintos."""
    if not categoricas:
        return [], []
    distintos = train.agg(*[F.countDistinct(c).alias(c) for c in categoricas]).first().asDict()
    ok = [c for c in categoricas if distintos[c] <= MAX_CATEGORIAS]
    fuera = [c for c in categoricas if distintos[c] > MAX_CATEGORIAS]
    return ok, fuera


def construir(silver: DataFrame) -> tuple:
    train = silver.filter(F.col("dataset") == "train")
    numericas, categoricas = clasificar_columnas(silver)
    categoricas, fuera = filtrar_alta_cardinalidad(train, categoricas)

    # Corte temporal: últimos DIAS_VALIDACION días del train = validación
    max_dia = train.agg(F.max("dia")).first()[0]
    corte = max_dia - DIAS_VALIDACION

    columnas_originales = numericas + categoricas
    df = silver.withColumn(
        # Cuenta los nulos de cada fila con un arreglo plano (más liviano que sumar 400 columnas)
        "n_nulos", F.size(F.filter(F.array(*[F.col(c).isNull() for c in columnas_originales]), lambda x: x))
    ).withColumn(
        "log_monto", F.log1p(F.col("TransactionAmt"))
    ).withColumn(
        "split",
        F.when(F.col("dataset") == "test", "test_kaggle")
        .when(F.col("dia") > corte, "validacion")
        .otherwise("entrenamiento"),
    )
    # Booleanos a 0/1 y nulos numéricos al centinela
    for c, tipo in df.dtypes:
        if c in numericas and tipo == "boolean":
            df = df.withColumn(c, F.col(c).cast("int"))
    df = df.fillna(CENTINELA, subset=numericas).fillna("missing", subset=categoricas)

    features_num = numericas + ["n_nulos", "log_monto"]
    df = df.select("TransactionID", "isFraud", "TransactionAmt", "split",
                   *[c for c in features_num if c != "TransactionAmt"], *categoricas)
    info = {
        "features_numericas": [c for c in features_num if c != "TransactionAmt"] + ["TransactionAmt"],
        "features_categoricas": categoricas,
        "categoricas_excluidas_por_cardinalidad": fuera,
        "corte_validacion_dia": int(corte),
        "dias_validacion": DIAS_VALIDACION,
        "centinela_nulos": CENTINELA,
    }
    return df, info


def run(spark: SparkSession) -> dict:
    log.info("=== GOLD: features para el modelo ===")
    silver = spark.read.format(TABLE_FORMAT).load(ruta(SILVER, "transacciones"))
    df, info = construir(silver)
    destino = ruta(GOLD, "features")
    df.write.format(TABLE_FORMAT).mode("overwrite").option("overwriteSchema", "true").partitionBy("split").save(destino)

    conteo = {r["split"]: r["count"] for r in spark.read.format(TABLE_FORMAT).load(destino).groupBy("split").count().collect()}
    info["filas_por_split"] = conteo
    META.mkdir(parents=True, exist_ok=True)
    (META / "gold_features.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Features: %d numéricas + %d categóricas (%d categóricas excluidas por tener más de %d valores)",
             len(info["features_numericas"]), len(info["features_categoricas"]),
             len(info["categoricas_excluidas_por_cardinalidad"]), MAX_CATEGORIAS)
    log.info("Validación por tiempo: días > %d  ·  filas por split: %s", info["corte_validacion_dia"], conteo)
    return info


if __name__ == "__main__":
    spark = get_spark("gold-features")
    run(spark)
    spark.stop()
