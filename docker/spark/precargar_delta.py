"""Se ejecuta una sola vez al construir la imagen Docker.

Arranca una sesión de Spark con Delta Lake para que Spark descargue los JAR
de Delta desde Maven y los deje guardados en la caché de la imagen (~/.ivy2).
Así el contenedor no depende de internet cada vez que se inicia.
"""
from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession

builder = (
    SparkSession.builder.appName("precarga-delta")
    .master("local[1]")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
)
spark = configure_spark_with_delta_pip(builder).getOrCreate()
spark.range(5).write.format("delta").mode("overwrite").save("/tmp/prueba_delta")
print("Delta Lake listo:", spark.read.format("delta").load("/tmp/prueba_delta").count(), "filas de prueba")
spark.stop()
