"""Exporta las tablas Gold a CSV para el dashboard de la Fase 3.

El lakehouse vive en un volumen de Docker; el dashboard (Streamlit) lee estos CSV,
así que se puede desarrollar en cualquier equipo sin Docker ni Spark.

Salida en dashboard/datos/ (visible desde Windows):
  * kpis_diarios.csv            por día: transacciones, fraudes, tasa y montos
  * fraude_por_segmento.csv     tasa y monto de fraude por producto, tarjeta, email, dispositivo y hora (P1)
  * kpis_dia_segmento.csv       conteos por día y segmento, para filtrar por fecha y categoría a la vez
  * curva_costo_umbral.csv      para cada umbral: matriz de confusión, Precision, Recall, alertas y costo (P3)
  * predicciones_validacion.csv una fila por transacción de los últimos 30 días, con la probabilidad
                                de fraude y sus datos de negocio (panel de alertas)
  * modelo_metricas.json, gold_kpis_resumen.json, eda_hallazgos.json

Uso:
    docker compose exec spark python scripts/exportar_gold.py
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pyspark.sql import functions as F  # noqa: E402

from src.config import GOLD, META, RAIZ, SILVER, TABLE_FORMAT, get_logger, get_spark, ruta  # noqa: E402

log = get_logger("exportar_gold")
DESTINO = RAIZ / "dashboard" / "datos"

# Datos de negocio que acompañan cada predicción en el panel de alertas
COLUMNAS_NEGOCIO = ["dia", "hora", "ProductCD", "card4", "card6", "P_email_proveedor", "DeviceType", "device_marca"]


def a_csv(df, nombre: str) -> None:
    pdf = df.toPandas()
    pdf.to_csv(DESTINO / f"{nombre}.csv", index=False, encoding="utf-8")
    log.info("%-28s %8s filas", f"{nombre}.csv", f"{len(pdf):,}")


def main() -> None:
    DESTINO.mkdir(parents=True, exist_ok=True)
    spark = get_spark("exportar-gold")
    leer = lambda capa, tabla: spark.read.format(TABLE_FORMAT).load(ruta(capa, tabla))  # noqa: E731
    try:
        a_csv(leer(GOLD, "kpis_diarios").orderBy("dia"), "kpis_diarios")
        a_csv(leer(GOLD, "fraude_por_segmento").orderBy("segmento", F.desc("tasa_fraude_pct")), "fraude_por_segmento")
        a_csv(leer(GOLD, "kpis_dia_segmento").orderBy("dia"), "kpis_dia_segmento")
        a_csv(leer(GOLD, "curva_costo_umbral").orderBy("umbral"), "curva_costo_umbral")

        silver = leer(SILVER, "transacciones")
        negocio = [c for c in COLUMNAS_NEGOCIO if c in silver.columns]
        predicciones = (
            leer(GOLD, "predicciones").filter(F.col("split") == "validacion").drop("split")
            .join(silver.select("TransactionID", *negocio), "TransactionID", "left")
            .orderBy("dia", F.desc("prob_fraude"))
        )
        a_csv(predicciones, "predicciones_validacion")
    finally:
        spark.stop()

    for nombre in ["modelo_metricas.json", "gold_kpis_resumen.json", "eda_hallazgos.json"]:
        if (META / nombre).exists():
            shutil.copy2(META / nombre, DESTINO / nombre)
    log.info("Datos del dashboard listos en %s", DESTINO)


if __name__ == "__main__":
    main()
