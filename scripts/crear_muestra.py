"""Crea una muestra pequeña de los CSV para desarrollar sin cargar 1,35 GB.

Toma una fracción aleatoria de las transacciones y SOLO las filas de identidad
que corresponden a esas transacciones, para que el left join siga teniendo sentido.
La muestra queda en data/sample/ con los mismos nombres de archivo.

Para usarla en el pipeline:  DATA_RAW=data/sample  LAKEHOUSE=data/lakehouse_sample

Uso:
    python scripts/crear_muestra.py --fraccion 0.05
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.config import DATA_RAW, RAIZ  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--fraccion", type=float, default=0.05, help="Fracción de transacciones (0.05 = 5 %%)")
parser.add_argument("--semilla", type=int, default=42)
args = parser.parse_args()

destino = RAIZ / "data" / "sample"
destino.mkdir(parents=True, exist_ok=True)

for dataset in ("train", "test"):
    tx = pd.read_csv(DATA_RAW / f"{dataset}_transaction.csv")
    tx = tx.sample(frac=args.fraccion, random_state=args.semilla).sort_values("TransactionDT")
    idn = pd.read_csv(DATA_RAW / f"{dataset}_identity.csv")
    idn = idn[idn["TransactionID"].isin(tx["TransactionID"])]
    tx.to_csv(destino / f"{dataset}_transaction.csv", index=False)
    idn.to_csv(destino / f"{dataset}_identity.csv", index=False)
    print(f"{dataset}: {len(tx):,} transacciones y {len(idn):,} filas de identidad -> {destino}")
