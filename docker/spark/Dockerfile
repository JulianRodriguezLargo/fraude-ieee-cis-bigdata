# Imagen propia del proyecto: Python 3.11 + Java 17 + PySpark 3.5 + Delta Lake 3.3 + JupyterLab.
# Se construye desde imágenes oficiales para no depender de imágenes de terceros
# (las de Bitnami y MinIO dejaron de publicarse gratis en 2025).
FROM python:3.11-slim-bookworm

# Java es obligatorio para Spark. En Debian 12 (bookworm), default-jre-headless es OpenJDK 17.
# procps (comando ps) lo usan los scripts internos de Spark.
RUN apt-get update \
    && apt-get install -y --no-install-recommends default-jre-headless procps \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/default-java \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/home/proyecto

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Descarga los JAR de Delta Lake al construir la imagen, para que el contenedor
# no necesite internet cada vez que arranca Spark.
COPY docker/spark/precargar_delta.py /tmp/precargar_delta.py
RUN python /tmp/precargar_delta.py

WORKDIR /home/proyecto

# 8888: JupyterLab · 4040: interfaz web de Spark (solo mientras hay un trabajo corriendo)
EXPOSE 8888 4040

CMD ["sh", "-c", "jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.token=${JUPYTER_TOKEN:-bigdata} --ServerApp.root_dir=/home/proyecto"]
