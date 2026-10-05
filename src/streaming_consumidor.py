"""Consumidor de streaming: Spark Structured Streaming lee de Kafka, califica con el modelo y guarda alertas.

Flujo:  Kafka (tópico transacciones) → Spark Structured Streaming → modelo GBT ya entrenado
        → gold/alertas_streaming (Delta)  +  dashboard/datos/streaming/alertas_stream.csv (para el dashboard)

Cada micro-lote (cada 2 segundos) se califica con el mismo PipelineModel del pipeline batch y el
umbral óptimo de la P3, así que batch y streaming usan exactamente la misma lógica.

Uso (dentro del contenedor, antes de lanzar el productor):
    python -m src.streaming_consumidor
Se detiene con Ctrl + C.
"""
import argparse
import json
import os
import shutil
import tempfile

from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StructField, StructType, TimestampType

from src.config import GOLD, LAKEHOUSE, META, RAIZ, TABLE_FORMAT, get_logger, get_spark, ruta

log = get_logger("streaming_consumidor")

SERVIDOR = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")
TOPICO = os.getenv("KAFKA_TOPICO", "transacciones")
SALIDA_CSV = RAIZ / "dashboard" / "datos" / "streaming" / "alertas_stream.csv"
TABLA = ruta(GOLD, "alertas_streaming")
NEGOCIO = ["ProductCD", "card4", "card6", "P_email_proveedor", "DeviceType", "device_marca"]


def esquema_mensaje(spark) -> StructType:
    """El JSON trae las columnas de gold/features más el día y la hora de envío."""
    base = spark.read.format(TABLE_FORMAT).load(ruta(GOLD, "features")).drop("split").schema
    return StructType(base.fields + [StructField("dia", IntegerType()), StructField("enviado_ts", TimestampType())])


def parsear(mensajes: DataFrame, esquema: StructType) -> DataFrame:
    """value (JSON) → columnas; kafka_ts es el momento en que Kafka recibió el mensaje."""
    return mensajes.select(
        F.from_json(F.col("value").cast("string"), esquema).alias("t"), F.col("timestamp").alias("kafka_ts")
    ).select("t.*", "kafka_ts")


def calificar(lote: DataFrame, modelo: PipelineModel, umbral: float) -> DataFrame:
    """Aplica el modelo a un micro-lote y deja solo lo que necesita el negocio."""
    pred = modelo.transform(lote).withColumn("prob_fraude", F.round(vector_to_array("probability")[1], 4))
    negocio = [c for c in NEGOCIO if c in pred.columns]
    return pred.select(
        "TransactionID", "dia", F.col("hora").cast("int").alias("hora"), "TransactionAmt", "isFraud", "prob_fraude",
        (F.col("prob_fraude") >= F.lit(umbral)).alias("alerta"), *negocio,
        "enviado_ts", "kafka_ts", F.current_timestamp().alias("calificado_ts"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Califica en tiempo real las transacciones que llegan a Kafka")
    parser.add_argument("--conservar", action="store_true", help="No borrar las alertas de corridas anteriores")
    args = parser.parse_args()

    umbral = json.loads((META / "modelo_metricas.json").read_text(encoding="utf-8"))["p3_umbral_optimo"]["umbral"]
    if not args.conservar:
        shutil.rmtree(TABLA, ignore_errors=True)
        SALIDA_CSV.unlink(missing_ok=True)
    SALIDA_CSV.parent.mkdir(parents=True, exist_ok=True)

    spark = get_spark("streaming-consumidor", kafka=True)
    modelo = PipelineModel.load(str(LAKEHOUSE / "modelos" / "gbt"))
    esquema = esquema_mensaje(spark)

    def escribir(lote: DataFrame, id_lote: int) -> None:
        if lote.isEmpty():
            return
        resultado = calificar(lote, modelo, umbral).cache()
        resultado.write.format(TABLE_FORMAT).mode("append").save(TABLA)
        pdf = resultado.toPandas()
        resultado.unpersist()
        pdf["latencia_s"] = (pdf["calificado_ts"] - pdf["enviado_ts"]).dt.total_seconds().round(2)
        pdf.to_csv(SALIDA_CSV, mode="a", header=not SALIDA_CSV.exists(), index=False)
        alertas = pdf[pdf["alerta"]]
        log.info("Lote %3d · %4d transacciones · %3d alertas · %3d fraudes reales entre las alertas · latencia media %.1f s",
                 id_lote, len(pdf), len(alertas), int(alertas["isFraud"].sum()), pdf["latencia_s"].mean())

    mensajes = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", SERVIDOR)
        .option("subscribe", TOPICO)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .load()
    )
    consulta = (
        parsear(mensajes, esquema).writeStream
        .foreachBatch(escribir)
        .option("checkpointLocation", tempfile.mkdtemp(prefix="checkpoint_streaming_"))
        .trigger(processingTime="2 seconds")
        .start()
    )
    log.info("=== CONSUMIDOR: escuchando el tópico '%s' en %s · umbral %.2f ===", TOPICO, SERVIDOR, umbral)
    log.info("Esperando transacciones... (ahora lanza el productor en otra ventana; Ctrl + C para detener)")
    try:
        consulta.awaitTermination()
    except KeyboardInterrupt:
        log.info("Deteniendo el streaming...")
    finally:
        consulta.stop()
        spark.stop()
        log.info(">>> Streaming detenido. Alertas en %s y %s", TABLA, SALIDA_CSV)


if __name__ == "__main__":
    main()
