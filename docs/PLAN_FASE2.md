# Plan de trabajo — Fase 2 (entrega: martes 6 de octubre)

**Lo que pide la guía:** repositorio Git (README + notebooks + scripts modulares + `requirements.txt`), pipeline ejecutable de punta a punta y capturas o logs que demuestren que corre. Los commits deben ser pequeños y frecuentes, y el historial debe reflejar trabajo distribuido.

**El código ya está escrito y probado** con datos de prueba que tienen el mismo esquema que IEEE-CIS. El trabajo de estos tres días es: correrlo con los datos reales, entender cada parte, interpretar los resultados, armar las evidencias y subir todo a Git.

## Quién es dueño de qué

Cada uno debe poder explicar las partes que sube a Git: el profesor puede preguntarle a cualquiera.

| Integrante | Archivos a su cargo | Debe poder explicar |
| --- | --- | --- |
| Julian | `docker-compose.yml`, `docker/`, `src/config.py`, `src/bronze_ingesta.py`, `src/pipeline.py`, `scripts/`, `README.md` | Por qué una imagen propia, cómo funciona Bronze append-only, cómo se corre todo |
| Samuel | `src/silver_limpieza.py`, `src/gold_kpis.py` | Cada paso de limpieza, el left join, por qué se decide con el train, las métricas de negocio (P1) |
| Samantha | `src/gold_features.py`, `src/modelo.py`, `src/graficos.py`, `notebooks/` | Por qué -999, class weights, validación por tiempo, cada métrica, el umbral óptimo (P2 y P3) |

## Sábado 3 de octubre

| Quién | Tarea | Listo cuando |
| --- | --- | --- |
| Julian | Instalar Docker en el portátil de 32 GB (README, pasos 1 a 6) | `verificar_entorno.py` dice ENTORNO LISTO |
| Julian | Corrida rápida: `python -m src.pipeline --fraccion 0.2 --iteraciones 30` | Termina sin errores y existe `_meta/modelo_metricas.json` |
| Julian | Crear el repositorio en GitHub (vacío) e invitar a Samuel y Samantha | Los tres tienen acceso |
| Samuel | Leer `silver_limpieza.py` y `gold_kpis.py` con el README al lado | Puede explicar cada función en voz alta |
| Samantha | Leer `gold_features.py` y `modelo.py` | Puede explicar cada decisión de la tabla del README |

**Si a medianoche Docker no funciona:** pasar al plan B (Colab, sección del README).

## Domingo 4 de octubre

| Quién | Tarea | Listo cuando |
| --- | --- | --- |
| Julian | Corrida completa: `python -m src.pipeline` (dejarla corriendo) | `pipeline.log` termina con "Pipeline completo" |
| Julian | Compartir `docs/evidencias/` (el pipeline copia ahí el log y los resúmenes JSON) | Los tres tienen las cifras reales |
| Samantha | En el equipo de Julian, o con JupyterLab en el suyo, correr `01_eda` y `02_modelo` completos | Los 11 gráficos están en `docs/evidencias/` |
| Samantha | Escribir la "Interpretación del equipo" en ambos notebooks | Texto con cifras reales, no genérico |
| Samuel | Llenar la tabla de Resultados del README con las cifras de `_meta/` | Tabla completa |
| Samuel | Revisar `gold/fraude_por_segmento`: ¿coincide con el EDA de la Fase 1 (producto C ~11,7 %)? | Confirmado o diferencia explicada |

## Lunes 5 de octubre

| Quién | Tarea | Listo cuando |
| --- | --- | --- |
| Los tres | Subir el código a Git en commits separados, cada uno desde su cuenta (orden abajo) | El historial muestra commits de los tres |
| Julian | Capturas: Docker corriendo, `verificar_entorno`, el final de `pipeline.log`, la Spark UI | Imágenes en `docs/evidencias/` |
| Samantha | Captura del notebook `02_modelo` con la tabla de métricas y la tabla de MLflow | Imágenes en `docs/evidencias/` |
| Todos | Repaso: cada uno explica sus archivos a los otros dos | Nadie depende de leer el código para responder |
| Julian | Revisión final: clonar el repositorio en otra carpeta y seguir el README desde cero | Funciona sin pasos ocultos |

## Martes 6 de octubre: entrega

## Orden sugerido de commits

El historial debe verse como un trabajo construido por partes, en este orden:

1. **Julian**: "Estructura del proyecto, Docker y configuración" (`docker-compose.yml`, `docker/`, `requirements.txt`, `.env.example`, `.gitignore`, `.gitattributes`, `src/__init__.py`, `src/config.py`)
2. **Julian**: "Capa Bronze: ingesta append-only con auditoría" (`src/bronze_ingesta.py`, `scripts/`)
3. **Samuel**: "Capa Silver: limpieza, join y tratamiento de nulos" (`src/silver_limpieza.py`)
4. **Samuel**: "Gold: KPIs de fraude por día y por segmento" (`src/gold_kpis.py`)
5. **Samantha**: "Gold: tabla de features para el modelo" (`src/gold_features.py`)
6. **Samantha**: "Modelo GBT con class weights, métricas y umbral óptimo" (`src/modelo.py`)
7. **Julian**: "Orquestador del pipeline completo" (`src/pipeline.py`)
8. **Samantha**: "Notebooks de EDA y modelo" (`src/graficos.py`, `notebooks/`)
9. **Samuel**: "Resultados de la corrida completa" (`README.md` con la tabla llena)
10. **Los tres**: evidencias, cada uno las suyas (`docs/evidencias/`)

## Checklist de entrega

- [ ] README que explica cómo instalar y correr todo desde cero
- [ ] `requirements.txt` con versiones fijadas
- [ ] El pipeline corre de punta a punta con una sola orden
- [ ] Limpieza (nulos, duplicados, formatos, outliers) con separación Bronze → Silver
- [ ] EDA: `describe()`, distribuciones, histogramas, boxplots, tendencia temporal y correlaciones, con hallazgos
- [ ] Al menos 2 métricas de negocio en Gold (`kpis_diarios`, `fraude_por_segmento`)
- [ ] Modelo en Spark MLlib con Recall, Precision, F1 y ROC-AUC, y el desbalance manejado
- [ ] Modelo registrado en MLflow e interpretado
- [ ] Capturas y `pipeline.log` como evidencia de que corre
- [ ] Commits de los tres integrantes
