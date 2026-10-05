"""MODELO: responde las preguntas P2 (¿qué tan bien predecimos el fraude?) y P3 (¿qué umbral cuesta menos?).

  * Modelo base: Regresión Logística, para tener contra qué comparar.
  * Modelo principal: GBT (Gradient-Boosted Trees) de Spark MLlib con class weights,
    porque solo ~3,5 % es fraude: cada fraude pesa tanto como ~27 transacciones legítimas.
  * Validación por tiempo: se entrena con los primeros días y se evalúa con los últimos 30.
  * Métricas: ROC-AUC, PR-AUC, Recall, Precision y F1. NUNCA accuracy.
  * P3: curva de costo según el umbral, con los costos supuestos de .env
    (COSTO_CONTRACARGO_USD y COSTO_REVISION_USD, pendientes de validar con el profesor).

Resultados:
  * _meta/modelo_metricas.json      métricas de ambos modelos, importancia de variables y umbral óptimo
  * gold/curva_costo_umbral         costo total para cada umbral (para el dashboard)
  * gold/predicciones               probabilidad de fraude de validación y test de Kaggle
  * modelos/gbt                     modelo entrenado (PipelineModel de Spark)
  * mlruns/                         experimentos registrados en MLflow

Uso:
    python -m src.modelo                    # datos completos
    python -m src.modelo --fraccion 0.2     # 20 % del entrenamiento, para probar rápido
"""
import argparse
import json
import os
import time

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pyspark.ml import Pipeline  # noqa: E402
from pyspark.ml.classification import GBTClassifier, LogisticRegression  # noqa: E402
from pyspark.ml.evaluation import BinaryClassificationEvaluator  # noqa: E402
from pyspark.ml.feature import StandardScaler, StringIndexer, VectorAssembler  # noqa: E402
from pyspark.ml.functions import vector_to_array  # noqa: E402
from pyspark.sql import DataFrame, SparkSession  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

from src.config import GOLD, LAKEHOUSE, META, TABLE_FORMAT, get_logger, get_spark, ruta  # noqa: E402

log = get_logger("modelo")

COSTO_CONTRACARGO = float(os.getenv("COSTO_CONTRACARGO_USD", "25"))  # supuesto: tarifa por contracargo
COSTO_REVISION = float(os.getenv("COSTO_REVISION_USD", "5"))        # supuesto: revisar una alerta
UMBRALES = np.round(np.arange(0.05, 0.96, 0.05), 2)


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
def cargar(spark: SparkSession, fraccion: float) -> tuple:
    info = json.loads((META / "gold_features.json").read_text(encoding="utf-8"))
    df = spark.read.format(TABLE_FORMAT).load(ruta(GOLD, "features"))
    entrenamiento = df.filter(F.col("split") == "entrenamiento")
    if fraccion < 1.0:
        # Se conservan TODOS los fraudes y se muestrean las legítimas, para no perder la clase minoritaria
        entrenamiento = entrenamiento.sampleBy("isFraud", fractions={0: fraccion, 1: 1.0}, seed=42)
    validacion = df.filter(F.col("split") == "validacion")
    test = df.filter(F.col("split") == "test_kaggle")
    return entrenamiento, validacion, test, info


def agregar_pesos(df: DataFrame) -> tuple:
    """Class weights: cada fraude pesa (legítimas / fraudes); cada legítima pesa 1."""
    conteo = {r["isFraud"]: r["count"] for r in df.groupBy("isFraud").count().collect()}
    peso = conteo.get(0, 1) / max(conteo.get(1, 1), 1)
    return df.withColumn("peso", F.when(F.col("isFraud") == 1, F.lit(peso)).otherwise(F.lit(1.0))), peso, conteo


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
def preprocesamiento(info: dict) -> tuple:
    cat = info["features_categoricas"]
    num = info["features_numericas"]
    indexadores = [StringIndexer(inputCol=c, outputCol=f"{c}__idx", handleInvalid="keep") for c in cat]
    entradas = num + [f"{c}__idx" for c in cat]
    ensamblador = VectorAssembler(inputCols=entradas, outputCol="features", handleInvalid="keep")
    return indexadores + [ensamblador], entradas


def pipeline_lr(etapas: list) -> Pipeline:
    escalador = StandardScaler(inputCol="features", outputCol="features_esc")
    lr = LogisticRegression(featuresCol="features_esc", labelCol="isFraud", weightCol="peso",
                            maxIter=30, regParam=0.01)
    return Pipeline(stages=etapas + [escalador, lr])


def pipeline_gbt(etapas: list, iteraciones: int) -> Pipeline:
    gbt = GBTClassifier(featuresCol="features", labelCol="isFraud", weightCol="peso",
                        maxIter=iteraciones, maxDepth=5, maxBins=64, stepSize=0.1,
                        subsamplingRate=0.8, seed=42)
    return Pipeline(stages=etapas + [gbt])


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------
def con_probabilidad(pred: DataFrame) -> DataFrame:
    return pred.withColumn("prob_fraude", vector_to_array("probability")[1])


def aucs(pred: DataFrame) -> dict:
    ev = BinaryClassificationEvaluator(labelCol="isFraud", rawPredictionCol="rawPrediction")
    return {
        "roc_auc": round(ev.setMetricName("areaUnderROC").evaluate(pred), 4),
        "pr_auc": round(ev.setMetricName("areaUnderPR").evaluate(pred), 4),
    }


def metricas_por_umbral(pdf: pd.DataFrame, dias: int) -> pd.DataFrame:
    """Para cada umbral: matriz de confusión, Precision, Recall, F1, alertas por día y costo total (P3)."""
    y = pdf["isFraud"].to_numpy()
    p = pdf["prob_fraude"].to_numpy()
    monto = pdf["TransactionAmt"].to_numpy()
    filas = []
    for u in UMBRALES:
        alerta = p >= u
        tp = int(np.sum(alerta & (y == 1)))
        fp = int(np.sum(alerta & (y == 0)))
        fn = int(np.sum(~alerta & (y == 1)))
        tn = int(np.sum(~alerta & (y == 0)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        # Costo: cada alerta se revisa; cada fraude no detectado cuesta su monto + el contracargo
        costo = (tp + fp) * COSTO_REVISION + float(np.sum(monto[~alerta & (y == 1)])) + fn * COSTO_CONTRACARGO
        filas.append({
            "umbral": float(u), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
            "alertas_por_dia": round((tp + fp) / max(dias, 1), 1),
            "costo_total_usd": round(costo, 2),
        })
    return pd.DataFrame(filas)


def importancia(modelo_gbt, entradas: list, top: int = 20) -> list:
    imp = modelo_gbt.stages[-1].featureImportances.toArray()
    orden = np.argsort(imp)[::-1][:top]
    return [{"variable": entradas[i].replace("__idx", ""), "importancia": round(float(imp[i]), 4)} for i in orden]


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------
def run(spark: SparkSession, fraccion: float = 1.0, iteraciones: int = 60, con_lr: bool = True) -> dict:
    inicio = time.time()
    log.info("=== MODELO: entrenamiento y evaluación (fracción de entrenamiento: %.0f %%) ===", fraccion * 100)
    entrenamiento, validacion, test, info = cargar(spark, fraccion)
    entrenamiento, peso, conteo = agregar_pesos(entrenamiento)
    entrenamiento = entrenamiento.cache()
    log.info("Entrenamiento: %s legítimas, %s fraudes -> peso de cada fraude: %.1f",
             f"{conteo.get(0, 0):,}", f"{conteo.get(1, 0):,}", peso)

    etapas, entradas = preprocesamiento(info)
    dias_val = info["dias_validacion"]
    resultados = {"peso_clase_fraude": round(peso, 2), "filas_entrenamiento": conteo,
                  "costos_supuestos": {"contracargo_usd": COSTO_CONTRACARGO, "revision_usd": COSTO_REVISION}}

    # --- Modelo base ---
    if con_lr:
        t = time.time()
        modelo_lr = pipeline_lr(etapas).fit(entrenamiento)
        pred_lr = con_probabilidad(modelo_lr.transform(validacion))
        resultados["regresion_logistica"] = aucs(pred_lr)
        pdf = pred_lr.select("isFraud", "prob_fraude", "TransactionAmt").toPandas()
        m05 = metricas_por_umbral(pdf, dias_val).set_index("umbral").loc[0.5]
        resultados["regresion_logistica"].update({k: float(m05[k]) for k in ("precision", "recall", "f1")})
        resultados["regresion_logistica"]["segundos"] = round(time.time() - t)
        log.info("Regresión Logística -> %s", resultados["regresion_logistica"])

    # --- Modelo principal ---
    t = time.time()
    modelo_gbt = pipeline_gbt(etapas, iteraciones).fit(entrenamiento)
    pred_val = con_probabilidad(modelo_gbt.transform(validacion))
    res_gbt = aucs(pred_val)
    pdf = pred_val.select("isFraud", "prob_fraude", "TransactionAmt").toPandas()
    curva = metricas_por_umbral(pdf, dias_val)
    m05 = curva.set_index("umbral").loc[0.5]
    res_gbt.update({k: float(m05[k]) for k in ("precision", "recall", "f1")})
    res_gbt["segundos"] = round(time.time() - t)
    res_gbt["iteraciones"] = iteraciones
    resultados["gbt"] = res_gbt
    log.info("GBT con class weights -> %s", res_gbt)

    # --- P2: criterio de éxito definido en la Fase 1 ---
    resultados["criterio_p2"] = {
        "meta": "ROC-AUC >= 0.85 y Recall >= 0.70 (umbral 0.5)",
        "cumple": bool(res_gbt["roc_auc"] >= 0.85 and res_gbt["recall"] >= 0.70),
    }

    # --- P3: umbral de menor costo ---
    mejor = curva.loc[curva["costo_total_usd"].idxmin()]
    sin_modelo = float(pdf.loc[pdf["isFraud"] == 1, "TransactionAmt"].sum() + (pdf["isFraud"] == 1).sum() * COSTO_CONTRACARGO)
    resultados["p3_umbral_optimo"] = {
        "umbral": float(mejor["umbral"]),
        "costo_total_usd": float(mejor["costo_total_usd"]),
        "costo_sin_modelo_usd": round(sin_modelo, 2),
        "ahorro_usd": round(sin_modelo - float(mejor["costo_total_usd"]), 2),
        "recall": float(mejor["recall"]), "precision": float(mejor["precision"]),
        "alertas_por_dia": float(mejor["alertas_por_dia"]),
        "periodo_validacion_dias": dias_val,
    }
    log.info("P3 -> umbral óptimo %.2f: costo %s USD vs %s USD sin modelo (ahorro %s USD en %d días)",
             mejor["umbral"], f"{mejor['costo_total_usd']:,.0f}", f"{sin_modelo:,.0f}",
             f"{sin_modelo - mejor['costo_total_usd']:,.0f}", dias_val)

    resultados["importancia_variables_top20"] = importancia(modelo_gbt, entradas)
    resultados["segundos_total"] = round(time.time() - inicio)

    # --- Guardar resultados ---
    spark.createDataFrame(curva).write.format(TABLE_FORMAT).mode("overwrite").option("overwriteSchema", "true") \
        .save(ruta(GOLD, "curva_costo_umbral"))
    umbral = float(mejor["umbral"])
    predicciones = con_probabilidad(modelo_gbt.transform(validacion.unionByName(test))) \
        .select("TransactionID", "split", "isFraud", "TransactionAmt", F.round("prob_fraude", 4).alias("prob_fraude")) \
        .withColumn("alerta", F.col("prob_fraude") >= F.lit(umbral))
    predicciones.write.format(TABLE_FORMAT).mode("overwrite").option("overwriteSchema", "true") \
        .partitionBy("split").save(ruta(GOLD, "predicciones"))
    modelo_gbt.write().overwrite().save(str(LAKEHOUSE / "modelos" / "gbt"))

    META.mkdir(parents=True, exist_ok=True)
    (META / "modelo_metricas.json").write_text(json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")
    registrar_mlflow(resultados, iteraciones, fraccion)
    entrenamiento.unpersist()
    log.info("Modelo guardado en %s · métricas en _meta/modelo_metricas.json", LAKEHOUSE / "modelos" / "gbt")
    return resultados


def registrar_mlflow(res: dict, iteraciones: int, fraccion: float) -> None:
    """Registra el experimento en MLflow (carpeta mlruns/ dentro del lakehouse)."""
    try:
        import mlflow
    except ImportError:
        log.warning("MLflow no está instalado: se omite el registro (las métricas quedan en el JSON)")
        return
    mlflow.set_tracking_uri((LAKEHOUSE / "mlruns").resolve().as_uri())
    mlflow.set_experiment("fraude-ieee-cis")
    with mlflow.start_run(run_name=f"gbt_{iteraciones}it_{int(fraccion * 100)}pct"):
        mlflow.log_params({"modelo": "GBTClassifier", "maxIter": iteraciones, "maxDepth": 5,
                           "fraccion_entrenamiento": fraccion, "peso_clase_fraude": res["peso_clase_fraude"],
                           "validacion": "ultimos 30 dias (temporal)"})
        mlflow.log_metrics({f"gbt_{k}": v for k, v in res["gbt"].items() if isinstance(v, (int, float))})
        if "regresion_logistica" in res:
            mlflow.log_metrics({f"lr_{k}": v for k, v in res["regresion_logistica"].items() if isinstance(v, (int, float))})
        mlflow.log_metrics({"p3_umbral_optimo": res["p3_umbral_optimo"]["umbral"],
                            "p3_ahorro_usd": res["p3_umbral_optimo"]["ahorro_usd"]})
        mlflow.log_artifact(str(META / "modelo_metricas.json"))
    log.info("Experimento registrado en MLflow (%s)", LAKEHOUSE / "mlruns")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fraccion", type=float, default=1.0, help="Fracción de legítimas para entrenar (los fraudes se usan todos)")
    parser.add_argument("--iteraciones", type=int, default=60, help="Árboles del GBT")
    parser.add_argument("--sin-lr", action="store_true", help="Omitir la Regresión Logística base")
    args = parser.parse_args()
    spark = get_spark("modelo-fraude")
    run(spark, args.fraccion, args.iteraciones, not args.sin_lr)
    spark.stop()
