"""Productor de streaming: envía a Kafka las transacciones de un día de validación, hora por hora.

Simula la llegada de pagos en tiempo real. Cada mensaje es una transacción en JSON con las mismas
columnas de gold/features (lo que necesita el modelo), más el día y la hora de envío.

Tópico: transacciones  ·  Servidor: kafka:9092 (servicio "kafka" de docker-compose, perfil "streaming")

Uso (dentro del contenedor, con el consumidor ya corriendo):
    python -m src.streaming_productor --dia 182 --pausa 4
"""
import argparse
import os
import time

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.config import GOLD, SILVER, TABLE_FORMAT, get_logger, get_spark, ruta

log = get_logger("streaming_productor")

SERVIDOR = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")
TOPICO = os.getenv("KAFKA_TOPICO", "transacciones")


def transacciones_del_dia(spark, dia: int) -> DataFrame:
    """Filas de gold/features del día pedido, en el orden en que ocurrieron."""
    features = spark.read.format(TABLE_FORMAT).load(ruta(GOLD, "features")).filter(F.col("split") == "validacion").drop("split")
    tiempo = spark.read.format(TABLE_FORMAT).load(ruta(SILVER, "transacciones")).select("TransactionID", "dia", "TransactionDT")
    return features.join(tiempo, "TransactionID").filter(F.col("dia") == dia).orderBy("TransactionDT").drop("TransactionDT")


def como_mensajes(df: DataFrame) -> DataFrame:
    """Formato de Kafka: key = TransactionID, value = la transacción completa en JSON."""
    df = df.withColumn("enviado_ts", F.current_timestamp())
    return df.select(F.col("TransactionID").cast("string").alias("key"), F.to_json(F.struct(*df.columns)).alias("value"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Envía transacciones de validación a Kafka, hora por hora")
    parser.add_argument("--dia", type=int, default=182, help="Día de validación a reproducir (153 a 182)")
    parser.add_argument("--pausa", type=float, default=4.0, help="Segundos entre una hora y la siguiente")
    args = parser.parse_args()

    spark = get_spark("streaming-productor", kafka=True)
    try:
        dia = transacciones_del_dia(spark, args.dia).cache()
        total = dia.count()
        if total == 0:
            log.error("El día %d no tiene transacciones de validación (usa un día entre 153 y 182)", args.dia)
            return
        log.info("=== PRODUCTOR: día %d · %s transacciones → tópico '%s' en %s ===", args.dia, f"{total:,}", TOPICO, SERVIDOR)
        enviadas = 0
        for hora in range(24):
            lote = dia.filter(F.col("hora") == hora)
            n = lote.count()
            if n:
                como_mensajes(lote).write.format("kafka").option("kafka.bootstrap.servers", SERVIDOR).option("topic", TOPICO).save()
                enviadas += n
            log.info("Hora %02d:00 · %4d transacciones enviadas (acumulado %s de %s)", hora, n, f"{enviadas:,}", f"{total:,}")
            time.sleep(args.pausa)
        log.info(">>> Día %d enviado completo: %s transacciones", args.dia, f"{enviadas:,}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
