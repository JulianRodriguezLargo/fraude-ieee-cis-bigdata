"""
EDA inicial - IEEE-CIS Fraud Detection
Saca las cifras reales para el Entregable 1 (Volume y Veracity).

Uso (desde la carpeta Proyecto-Final-BigData):
    py -m pip install pandas
    py eda_inicial.py

Genera: resultados_eda.txt (y lo muestra en pantalla).
Lee un archivo a la vez para no llenar la RAM (~2-3 GB por archivo grande).
"""
import os
import gc
import pandas as pd

RAW = os.path.join("data", "raw")
ARCHIVOS = ["train_transaction.csv", "train_identity.csv",
            "test_transaction.csv", "test_identity.csv"]

lineas = []


def out(texto=""):
    print(texto)
    lineas.append(str(texto))


out("=" * 70)
out("EDA INICIAL - IEEE-CIS FRAUD DETECTION")
out("=" * 70)

# ---------------------------------------------------------------
# 1. VOLUME: filas, columnas, tamaño en disco y en memoria
# ---------------------------------------------------------------
out("\n[1] VOLUME")
out(f"{'Archivo':<25}{'Filas':>12}{'Columnas':>10}{'Disco (MB)':>12}{'RAM (MB)':>12}")
resumen = {}
for nombre in ARCHIVOS:
    ruta = os.path.join(RAW, nombre)
    df = pd.read_csv(ruta)
    disco = os.path.getsize(ruta) / 1e6
    ram = df.memory_usage(deep=True).sum() / 1e6
    resumen[nombre] = (len(df), df.shape[1])
    out(f"{nombre:<25}{len(df):>12,}{df.shape[1]:>10}{disco:>12,.1f}{ram:>12,.1f}")
    del df
    gc.collect()

total_disco = sum(os.path.getsize(os.path.join(RAW, a)) for a in ARCHIVOS) / 1e9
total_tx = resumen["train_transaction.csv"][0] + resumen["test_transaction.csv"][0]
out(f"\nTotal en disco (4 CSV): {total_disco:.2f} GB")
out(f"Total transacciones (train + test): {total_tx:,}")

# ---------------------------------------------------------------
# 2. VERACITY sobre el TRAIN (transaction + identity)
# ---------------------------------------------------------------
out("\n[2] VERACITY (train)")
tx = pd.read_csv(os.path.join(RAW, "train_transaction.csv"))
idn = pd.read_csv(os.path.join(RAW, "train_identity.csv"))

# Desbalance
fraudes = int(tx["isFraud"].sum())
out(f"Fraudes: {fraudes:,} de {len(tx):,} = {fraudes / len(tx) * 100:.2f} %")
out(f"Legitimas: {(1 - fraudes / len(tx)) * 100:.2f} %")

# Duplicados
dup = len(tx) - tx["TransactionID"].nunique()
out(f"TransactionID duplicados: {dup:,}")

# Cobertura de identidad
con_id = tx["TransactionID"].isin(idn["TransactionID"]).mean() * 100
out(f"Transacciones con identidad: {con_id:.1f} %")

# Join (left) y nulos
df = tx.merge(idn, on="TransactionID", how="left")
del tx, idn
gc.collect()
out(f"Columnas tras el join: {df.shape[1]}")

nulos = df.isna().mean().sort_values(ascending=False) * 100
out(f"% total de celdas nulas: {df.isna().values.mean() * 100:.1f} %")
out(f"Columnas con >90 % nulos: {(nulos > 90).sum()}")
out(f"Columnas con >50 % nulos: {(nulos > 50).sum()}")
out(f"Columnas sin nulos: {(nulos == 0).sum()}")
out("Top 10 columnas con mas nulos:")
for col, pct in nulos.head(10).items():
    out(f"   {col:<15}{pct:6.1f} %")

# Monto (outliers)
amt = df["TransactionAmt"]
out("\nTransactionAmt (USD):")
out(f"   min {amt.min():,.2f} | mediana {amt.median():,.2f} | media {amt.mean():,.2f} "
    f"| p99 {amt.quantile(0.99):,.2f} | max {amt.max():,.2f}")

# Rango temporal (TransactionDT en segundos)
dt = df["TransactionDT"]
dias = (dt.max() - dt.min()) / 86400
out(f"\nTransactionDT: {dt.min():,} a {dt.max():,} segundos = {dias:.0f} dias")
out(f"Promedio: {len(df) / dias:,.0f} transacciones/dia")

# Tasa de fraude por segmento (para P1 y Value)
out("\nTasa de fraude por segmento:")
for col in ["ProductCD", "card4", "card6", "DeviceType"]:
    t = df.groupby(col)["isFraud"].agg(["count", "mean"]).sort_values("mean", ascending=False)
    out(f"  {col}:")
    for k, r in t.iterrows():
        out(f"     {str(k):<18} n={int(r['count']):>8,}  fraude={r['mean'] * 100:5.2f} %")

monto_fraude = df.loc[df["isFraud"] == 1, "TransactionAmt"].sum()
out(f"\nMonto total defraudado (train): {monto_fraude:,.2f} USD")

with open("resultados_eda.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lineas))
out("\nListo. Resultados guardados en resultados_eda.txt")
