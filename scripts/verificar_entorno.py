"""Verifica que el entorno funcione: Spark arranca, Delta Lake escribe y lee, y los datos están en su sitio.

Uso (dentro del contenedor):
    python scripts/verificar_entorno.py
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_RAW, FUENTES_IEEE, LAKEHOUSE, TABLE_FORMAT, get_spark  # noqa: E402

ok = True
print("=" * 60)
print("VERIFICACIÓN DEL ENTORNO")
print("=" * 60)

# 1. Spark
spark = get_spark("verificar-entorno")
print(f"[OK] Spark {spark.version} arrancó  ·  memoria del driver: {spark.conf.get('spark.driver.memory')}")

# 2. Escritura y lectura en el formato del lakehouse (Delta)
prueba = LAKEHOUSE / "_prueba_entorno"
spark.range(1000).write.format(TABLE_FORMAT).mode("overwrite").save(str(prueba))
n = spark.read.format(TABLE_FORMAT).load(str(prueba)).count()
shutil.rmtree(prueba, ignore_errors=True)
if n == 1000:
    print(f"[OK] Escritura y lectura en formato {TABLE_FORMAT} ({n} filas de prueba)")
else:
    ok = False
    print(f"[ERROR] Se esperaban 1000 filas y se leyeron {n}")

# 3. Datos crudos
for archivo in FUENTES_IEEE:
    ruta = DATA_RAW / archivo
    if ruta.exists():
        print(f"[OK] {archivo:<24} {ruta.stat().st_size / 1e6:8.1f} MB")
    else:
        ok = False
        print(f"[FALTA] {archivo:<21} cópialo en data/raw/ (ver README, paso 4)")

spark.stop()
print("=" * 60)
print("ENTORNO LISTO" if ok else "HAY PENDIENTES: revisa las líneas marcadas arriba")
sys.exit(0 if ok else 1)
