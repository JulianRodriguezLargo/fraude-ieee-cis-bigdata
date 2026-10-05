"""Capa GOLD (KPIs): métricas de negocio que alimentan el dashboard y responden la pregunta P1.

Tablas que produce (solo con el train, que es el que tiene la etiqueta isFraud):
  * gold/kpis_diarios          -> por día: transacciones, fraudes, tasa de fraude, monto total y defraudado
  * gold/fraude_por_segmento   -> tasa y monto de fraude por cada valor de cada segmento
                                  (producto, tarjeta, email, dispositivo, hora)
  * gold/kpis_dia_segmento     -> conteos por día y combinación de producto, tarjeta, dispositivo y email:
                                  permite que el dashboard filtre por fecha y categoría a la vez
  * _meta/gold_kpis_resumen.json -> KPIs globales

Uso:
    python -m src.gold_kpis
"""
import json
from functools import reduce

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.config import GOLD, META, SILVER, TABLE_FORMAT, get_logger, get_spark, ruta

log = get_logger("gold_kpis")

# Segmentos de la pregunta P1. Solo se usan los que existan en Silver.
SEGMENTOS = ["ProductCD", "card4", "card6", "P_email_proveedor", "DeviceType", "device_marca", "hora", "dia_semana"]
MIN_TRANSACCIONES = 100  # valores con menos transacciones no son confiables para una tasa
# Dimensiones del cubo diario del dashboard (pocas y de baja cardinalidad, para que la tabla quede pequeña)
DIMENSIONES_DASHBOARD = ["ProductCD", "card6", "DeviceType", "P_email_proveedor"]


def agregados(df: DataFrame) -> list:
    """Las mismas métricas para cualquier agrupación."""
    return [
        F.count("*").alias("transacciones"),
        F.sum("isFraud").alias("fraudes"),
        F.round(100 * F.avg("isFraud"), 3).alias("tasa_fraude_pct"),
        F.round(F.sum("TransactionAmt"), 2).alias("monto_total_usd"),
        F.round(F.sum(F.when(F.col("isFraud") == 1, F.col("TransactionAmt")).otherwise(0)), 2).alias("monto_fraude_usd"),
    ]


def kpis_diarios(train: DataFrame) -> DataFrame:
    return train.groupBy("dia").agg(*agregados(train)).orderBy("dia")


def fraude_por_segmento(train: DataFrame) -> DataFrame:
    """Formato largo (segmento, valor, métricas): fácil de filtrar en el dashboard."""
    partes = []
    for seg in SEGMENTOS:
        if seg not in train.columns:
            continue
        partes.append(
            train.withColumn("valor", F.coalesce(F.col(seg).cast("string"), F.lit("(sin dato)")))
            .groupBy("valor").agg(*agregados(train))
            .withColumn("segmento", F.lit(seg))
        )
    df = reduce(lambda a, b: a.unionByName(b), partes)
    return (
        df.filter(F.col("transacciones") >= MIN_TRANSACCIONES)
        .select("segmento", "valor", "transacciones", "fraudes", "tasa_fraude_pct", "monto_total_usd", "monto_fraude_usd")
        .orderBy("segmento", F.desc("tasa_fraude_pct"))
    )


def kpis_dia_segmento(train: DataFrame) -> DataFrame:
    """Conteos y montos por día y combinación de segmentos.

    Guarda sumas (no tasas): el dashboard suma las filas que quedan después de filtrar
    y recién ahí calcula la tasa de fraude, así el resultado es correcto con cualquier filtro.
    """
    dims = [c for c in DIMENSIONES_DASHBOARD if c in train.columns]
    df = train
    for c in dims:
        df = df.withColumn(c, F.coalesce(F.col(c).cast("string"), F.lit("(sin dato)")))
    return (
        df.groupBy("dia", *dims)
        .agg(F.count("*").alias("transacciones"), F.sum("isFraud").alias("fraudes"),
             F.round(F.sum("TransactionAmt"), 2).alias("monto_total_usd"),
             F.round(F.sum(F.when(F.col("isFraud") == 1, F.col("TransactionAmt")).otherwise(0)), 2).alias("monto_fraude_usd"))
        .orderBy("dia", *dims)
    )


def run(spark: SparkSession) -> dict:
    log.info("=== GOLD: KPIs de negocio ===")
    silver = spark.read.format(TABLE_FORMAT).load(ruta(SILVER, "transacciones"))
    train = silver.filter(F.col("dataset") == "train").cache()

    diarios = kpis_diarios(train)
    diarios.write.format(TABLE_FORMAT).mode("overwrite").option("overwriteSchema", "true").save(ruta(GOLD, "kpis_diarios"))

    segmentos = fraude_por_segmento(train)
    segmentos.write.format(TABLE_FORMAT).mode("overwrite").option("overwriteSchema", "true").save(ruta(GOLD, "fraude_por_segmento"))

    cubo = kpis_dia_segmento(train)
    cubo.write.format(TABLE_FORMAT).mode("overwrite").option("overwriteSchema", "true").save(ruta(GOLD, "kpis_dia_segmento"))

    g = train.agg(*agregados(train)).first()
    resumen = {
        "transacciones_train": g["transacciones"],
        "fraudes": g["fraudes"],
        "tasa_fraude_pct": g["tasa_fraude_pct"],
        "monto_total_usd": g["monto_total_usd"],
        "monto_fraude_usd": g["monto_fraude_usd"],
        "dias": diarios.count(),
    }
    # Segmento más riesgoso de cada tipo (respuesta inicial a P1)
    top = (
        spark.read.format(TABLE_FORMAT).load(ruta(GOLD, "fraude_por_segmento"))
        .groupBy("segmento").agg(F.max_by("valor", "tasa_fraude_pct").alias("valor_mas_riesgoso"),
                                 F.max("tasa_fraude_pct").alias("tasa_pct"))
        .collect()
    )
    resumen["segmento_mas_riesgoso"] = {r["segmento"]: {"valor": r["valor_mas_riesgoso"], "tasa_fraude_pct": r["tasa_pct"]} for r in top}
    train.unpersist()

    META.mkdir(parents=True, exist_ok=True)
    (META / "gold_kpis_resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("KPIs: %s transacciones · %s fraudes (%.2f %%) · %s USD defraudados",
             f"{resumen['transacciones_train']:,}", f"{resumen['fraudes']:,}", resumen["tasa_fraude_pct"], f"{resumen['monto_fraude_usd']:,.2f}")
    for seg, v in resumen["segmento_mas_riesgoso"].items():
        log.info("  más riesgoso en %-18s %-14s %6.2f %%", seg, v["valor"], v["tasa_fraude_pct"])
    return resumen


if __name__ == "__main__":
    spark = get_spark("gold-kpis")
    run(spark)
    spark.stop()
