"""Dashboard de detección de fraude en pagos en línea (Fase 3).

Lee las tablas Gold exportadas a CSV por scripts/exportar_gold.py (carpeta dashboard/datos/)
y responde las tres preguntas de negocio:
  P1  ¿Qué patrones caracterizan el fraude?          -> pestaña "Panorama del fraude"
  P2  ¿Qué tan bien lo predice el modelo?             -> pestaña "Modelo"
  P3  ¿Qué umbral minimiza el costo de los errores?   -> pestaña "Umbral y costo"
y agrega un panel de monitoreo que simula la llegada de transacciones en vivo.

Ejecutar desde la raíz del repositorio (ahí está el tema en .streamlit/config.toml):
    pip install -r dashboard/requirements.txt
    streamlit run dashboard/app.py
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATOS = Path(__file__).resolve().parent / "datos"
REPO = "https://github.com/JulianRodriguezLargo/fraude-ieee-cis-bigdata"

# Paleta validada para daltonismo: la misma de los notebooks (azul = legítima, rojo = fraude)
LEGITIMA = "#2a78d6"
FRAUDE = "#e34948"
MODELO_2 = "#eb6834"
TEXTO = "#0b0b0b"
TEXTO_2 = "#52514e"
REJILLA = "#e4e3dd"
REFERENCIA = "#52514e"
AZULES = [[0, "#f1f6fd"], [0.5, "#7fb0ea"], [1, "#1c5aa6"]]  # secuencial de un solo tono para la matriz

# Nombres de negocio para las columnas técnicas
NOMBRES = {
    "ProductCD": "Producto", "card6": "Tipo de tarjeta", "DeviceType": "Dispositivo",
    "P_email_proveedor": "Correo del comprador", "card4": "Red de la tarjeta",
    "device_marca": "Marca del dispositivo", "hora": "Hora", "dia_semana": "Día de la semana",
}
FILTROS = ["ProductCD", "card6", "DeviceType", "P_email_proveedor"]
MIN_TRANSACCIONES = 100  # una tasa con menos transacciones no es confiable

st.set_page_config(page_title="Fraude en pagos · IEEE-CIS", page_icon="🛡️", layout="wide")


# ---------------------------------------------------------------- formato (español: 1.234,5)
def entero(n: float) -> str:
    return f"{n:,.0f}".replace(",", ".")


def usd(n: float) -> str:
    return "US$ " + entero(n)


def corto(n: float) -> str:
    """Montos grandes en las tarjetas (la moneda va en la etiqueta): 3,08 M en vez de 3.083.845."""
    if abs(n) >= 1e6:
        return f"{n / 1e6:.2f}".replace(".", ",") + " M"
    if abs(n) >= 1e4:
        return f"{n / 1e3:.0f}" + " mil"
    return entero(n)


def pct(x: float, dec: int = 2) -> str:
    return f"{x:.{dec}f}".replace(".", ",") + " %"


def dec(x: float, d: int = 3) -> str:
    return f"{x:.{d}f}".replace(".", ",")


def estilo(fig: go.Figure, alto: int = 320, leyenda: bool = False) -> go.Figure:
    """Estilo común: rejilla suave, texto en tinta neutra, separadores en español."""
    fig.update_layout(
        height=alto, margin=dict(l=8, r=8, t=70 if leyenda else 40, b=8), separators=",.",
        title=dict(y=0.98, yanchor="top", x=0, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, Helvetica, Arial, sans-serif", size=13, color=TEXTO_2),
        title_font=dict(size=15, color=TEXTO), hoverlabel=dict(bgcolor="white", font_size=13),
        showlegend=leyenda,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, title=None),
    )
    fig.update_xaxes(showgrid=False, linecolor=REJILLA, ticks="", zeroline=False)
    fig.update_yaxes(gridcolor=REJILLA, zeroline=False, ticks="")
    return fig


def grafico(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


# ---------------------------------------------------------------- datos
@st.cache_data
def cargar() -> dict:
    necesarios = ["kpis_dia_segmento.csv", "fraude_por_segmento.csv", "predicciones_validacion.csv",
                  "modelo_metricas.json", "gold_kpis_resumen.json"]
    faltan = [f for f in necesarios if not (DATOS / f).exists()]
    if faltan:
        return {"faltan": faltan}
    pred = pd.read_csv(DATOS / "predicciones_validacion.csv")
    for c in FILTROS:
        if c in pred.columns:
            pred[c] = pred[c].fillna("(sin dato)").astype(str)
    cubo = pd.read_csv(DATOS / "kpis_dia_segmento.csv", dtype={c: str for c in FILTROS})
    return {
        "cubo": cubo,
        "segmentos": pd.read_csv(DATOS / "fraude_por_segmento.csv", dtype={"valor": str}),
        "pred": pred,
        "metricas": json.loads((DATOS / "modelo_metricas.json").read_text(encoding="utf-8")),
        "resumen": json.loads((DATOS / "gold_kpis_resumen.json").read_text(encoding="utf-8")),
    }


@st.cache_data
def curva_costo(prob: np.ndarray, y: np.ndarray, monto: np.ndarray, revision: float, contracargo: float,
                dias: int) -> pd.DataFrame:
    """Para cada umbral: matriz de confusión, Recall, Precision, alertas por día y costo total.

    Costo = alertas revisadas x costo de revisión + fraudes no detectados x (su monto + contracargo).
    Es la misma fórmula de src/modelo.py, recalculada aquí para poder cambiar los costos supuestos.
    """
    filas = []
    for u in np.round(np.arange(0.05, 0.951, 0.05), 2):  # misma rejilla que src/modelo.py
        alerta = prob >= u
        tp = int(np.sum(alerta & (y == 1)))
        fp = int(np.sum(alerta & (y == 0)))
        fn = int(np.sum(~alerta & (y == 1)))
        tn = int(np.sum(~alerta & (y == 0)))
        costo = (tp + fp) * revision + float(monto[~alerta & (y == 1)].sum()) + fn * contracargo
        filas.append({"umbral": u, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                      "recall": tp / (tp + fn) if tp + fn else 0.0,
                      "precision": tp / (tp + fp) if tp + fp else 0.0,
                      "alertas_por_dia": (tp + fp) / dias, "costo": costo})
    return pd.DataFrame(filas)


d = cargar()
if "faltan" in d:
    st.title("Detección de fraude en pagos en línea")
    st.error("Faltan los datos del dashboard: " + ", ".join(d["faltan"]))
    st.markdown("Genera los archivos con el pipeline y expórtalos desde la raíz del repositorio:\n\n"
                "```\ndocker compose exec spark python scripts/exportar_gold.py\n```")
    st.stop()

cubo, segmentos, pred, metricas, resumen = d["cubo"], d["segmentos"], d["pred"], d["metricas"], d["resumen"]
TASA_GLOBAL = 100 * resumen["fraudes"] / resumen["transacciones_train"]
DIA_MIN, DIA_MAX = int(cubo["dia"].min()), int(cubo["dia"].max())
DIAS_VALIDACION = sorted(pred["dia"].unique())
INICIO_VALIDACION = int(min(DIAS_VALIDACION))

# ---------------------------------------------------------------- filtros (barra lateral)
with st.sidebar:
    st.header("Filtros")
    rango = st.slider("Rango de días", DIA_MIN, DIA_MAX, (DIA_MIN, DIA_MAX),
                      help="Día 1 es la fecha de referencia de Vesta, que no es pública: los días son relativos.")
    seleccion = {}
    for c in FILTROS:
        if c in cubo.columns:
            opciones = sorted(cubo[c].unique(), key=lambda v: (v == "(sin dato)", v))
            seleccion[c] = st.multiselect(NOMBRES[c], opciones, placeholder="Todos")
    st.caption("Los filtros aplican al panorama del fraude y al monitoreo de alertas. "
               "Dejar un filtro vacío equivale a incluir todos los valores.")
    st.divider()
    st.caption(f"Datos: IEEE-CIS Fraud Detection (Vesta Corporation), {entero(resumen['transacciones_train'])} "
               f"transacciones etiquetadas de {resumen['dias']} días.  \n[Repositorio del proyecto]({REPO})")


def aplicar_filtros(df: pd.DataFrame) -> pd.DataFrame:
    for c, valores in seleccion.items():
        if valores and c in df.columns:
            df = df[df[c].isin(valores)]
    return df


filtrado = aplicar_filtros(cubo[cubo["dia"].between(*rango)])
hay_filtros = any(seleccion.values()) or rango != (DIA_MIN, DIA_MAX)

# ---------------------------------------------------------------- encabezado y KPIs
st.title("Detección de fraude en pagos en línea")
st.markdown(
    "Transacciones de comercio electrónico de **Vesta Corporation** (dataset IEEE-CIS), procesadas con "
    "Spark y Delta Lake en una arquitectura Medallion. Este tablero responde **qué caracteriza al fraude**, "
    "**qué tan bien lo detecta el modelo** y **con qué umbral conviene generar alertas** para que el "
    "equipo de Riesgo decida qué transacciones revisar."
)

tx = int(filtrado["transacciones"].sum())
fr = int(filtrado["fraudes"].sum())
monto_total = float(filtrado["monto_total_usd"].sum())
monto_fraude = float(filtrado["monto_fraude_usd"].sum())
tasa = 100 * fr / tx if tx else 0.0

diario = filtrado.groupby("dia", as_index=False)[["transacciones", "fraudes", "monto_total_usd", "monto_fraude_usd"]].sum()
diario["tasa"] = 100 * diario["fraudes"] / diario["transacciones"]
diario["tasa_7d"] = 100 * diario["fraudes"].rolling(7, min_periods=1).sum() / \
    diario["transacciones"].rolling(7, min_periods=1).sum()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Transacciones", entero(tx), border=True, chart_data=diario["transacciones"].tolist(), chart_type="bar")
k2.metric("Fraudes", entero(fr), border=True, chart_data=diario["fraudes"].tolist(), chart_type="bar")
k3.metric("Tasa de fraude", pct(tasa),
          delta=(f"{tasa - TASA_GLOBAL:+.2f}".replace(".", ",") + " pp vs. promedio" if hay_filtros else None),
          delta_color="inverse", border=True, chart_data=diario["tasa_7d"].round(2).tolist(),
          help=f"Promedio de todo el periodo: {pct(TASA_GLOBAL)}. La línea es el promedio móvil de 7 días.")
k4.metric("Defraudado (US$)", corto(monto_fraude), border=True, help=usd(monto_fraude),
          chart_data=diario["monto_fraude_usd"].round(0).tolist(), chart_type="bar")
k5.metric("Del monto total", pct(100 * monto_fraude / monto_total if monto_total else 0), border=True,
          help="Porcentaje del dinero transado que corresponde a fraude",
          chart_data=(100 * diario["monto_fraude_usd"].rolling(7, min_periods=1).sum()
                      / diario["monto_total_usd"].rolling(7, min_periods=1).sum()).round(2).tolist())

if tx == 0:
    st.warning("No hay transacciones con esta combinación de filtros.")
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Panorama del fraude (P1)", "Modelo (P2)",
                                        "Umbral y costo (P3)", "Monitoreo de alertas", "Streaming en vivo"])

# ================================================================ P1
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Bar(x=diario["dia"], y=diario["transacciones"], marker_color=LEGITIMA,
                               hovertemplate="Día %{x}<br>%{y:,.0f} transacciones<extra></extra>"))
        fig.update_layout(title="Transacciones por día", bargap=0.15)
        fig.update_xaxes(title="Día", range=[rango[0] - 1, rango[1] + 1])
        grafico(estilo(fig))
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=diario["dia"], y=diario["tasa"], mode="lines", name="Diaria",
                                 line=dict(color=FRAUDE, width=1), opacity=0.35,
                                 hovertemplate="Día %{x}<br>%{y:.2f} %<extra>Diaria</extra>"))
        fig.add_trace(go.Scatter(x=diario["dia"], y=diario["tasa_7d"], mode="lines", name="Promedio móvil de 7 días",
                                 line=dict(color=FRAUDE, width=2.5),
                                 hovertemplate="Día %{x}<br>%{y:.2f} %<extra>7 días</extra>"))
        fig.add_hline(y=TASA_GLOBAL, line=dict(color=REFERENCIA, dash="dash", width=1),
                      annotation_text=f"Promedio {pct(TASA_GLOBAL)}", annotation_position="top left",
                      annotation_font_color=TEXTO_2)
        if rango[1] >= INICIO_VALIDACION:
            fig.add_vrect(x0=max(INICIO_VALIDACION, rango[0]) - 0.5, x1=rango[1] + 0.5, fillcolor=REJILLA,
                          opacity=0.5, line_width=0, layer="below",
                          annotation_text="Validación", annotation_position="top right",
                          annotation_font_color=TEXTO_2)
        fig.update_layout(title="Tasa de fraude por día (%)")
        fig.update_xaxes(title="Día", range=[rango[0] - 1, rango[1] + 1])
        grafico(estilo(fig, leyenda=True))

    st.subheader("¿Dónde se concentra el fraude?")
    disponibles = [c for c in FILTROS if c in filtrado.columns]
    dim = st.segmented_control("Ver por", disponibles, format_func=lambda c: NOMBRES[c],
                               default=disponibles[0], key="dimension")
    dim = dim or disponibles[0]
    por_seg = filtrado.groupby(dim, as_index=False)[["transacciones", "fraudes", "monto_fraude_usd"]].sum()
    por_seg = por_seg[por_seg["transacciones"] >= MIN_TRANSACCIONES]
    por_seg["tasa"] = 100 * por_seg["fraudes"] / por_seg["transacciones"]
    por_seg = por_seg.sort_values("tasa").tail(12)

    c1, c2 = st.columns([3, 2])
    with c1:
        fig = go.Figure(go.Bar(
            x=por_seg["tasa"], y=por_seg[dim], orientation="h", marker_color=FRAUDE,
            text=[pct(v) for v in por_seg["tasa"]], textposition="outside", textfont_color=TEXTO_2,
            customdata=np.stack([por_seg["transacciones"], por_seg["fraudes"], por_seg["monto_fraude_usd"]], axis=1),
            hovertemplate="<b>%{y}</b><br>Tasa de fraude: %{x:.2f} %<br>Transacciones: %{customdata[0]:,.0f}"
                          "<br>Fraudes: %{customdata[1]:,.0f}<br>Monto defraudado: US$ %{customdata[2]:,.0f}<extra></extra>"))
        fig.add_vline(x=tasa, line=dict(color=REFERENCIA, dash="dash", width=1),
                      annotation_text=f"Selección {pct(tasa)}", annotation_position="bottom right",
                      annotation_font_color=TEXTO_2)
        fig.update_layout(title=f"Tasa de fraude por {NOMBRES[dim].lower()} (%)", bargap=0.3)
        fig.update_xaxes(range=[0, por_seg["tasa"].max() * 1.25 if len(por_seg) else 1])
        grafico(estilo(fig, alto=max(260, 34 * len(por_seg) + 70)))
    with c2:
        if len(por_seg):
            top = por_seg.iloc[-1]
            veces = top["tasa"] / tasa if tasa else 0
            st.markdown(f"#### Lectura rápida")
            st.markdown(
                f"- **{NOMBRES[dim]} = {top[dim]}** tiene la tasa más alta: **{pct(top['tasa'])}**, "
                f"{dec(veces, 1)} veces la de la selección.\n"
                f"- Ese segmento concentra **{usd(top['monto_fraude_usd'])}** defraudados.\n"
                f"- Solo se muestran valores con al menos {MIN_TRANSACCIONES} transacciones, para que la tasa sea confiable."
            )
            st.info("**Decisión para Riesgo:** los segmentos a la derecha de la línea punteada son candidatos a "
                    "revisión adicional o a reglas específicas mientras el modelo madura.")

    horas = segmentos[segmentos["segmento"] == "hora"].copy()
    if len(horas):
        horas["hora_n"] = horas["valor"].astype(float).astype(int)
        horas = horas.sort_values("hora_n")
        fig = go.Figure(go.Bar(
            x=horas["hora_n"], y=horas["tasa_fraude_pct"], marker_color=FRAUDE,
            customdata=horas["transacciones"],
            hovertemplate="Hora %{x}<br>Tasa de fraude: %{y:.2f} %<br>%{customdata:,.0f} transacciones<extra></extra>"))
        fig.add_hline(y=TASA_GLOBAL, line=dict(color=REFERENCIA, dash="dash", width=1))
        fig.update_layout(title="Tasa de fraude por hora del día (%) · todo el periodo", bargap=0.2)
        fig.update_xaxes(title="Hora (relativa a la referencia de Vesta, no es la hora local)", dtick=1)
        grafico(estilo(fig, alto=280))

# ================================================================ P2
with tab2:
    gbt = metricas["gbt"]
    lr = metricas.get("regresion_logistica")
    base_pr = 100 * pred["isFraud"].mean()
    st.markdown(f"El modelo se entrenó con los primeros {INICIO_VALIDACION - 1} días y se evalúa con los **últimos {len(DIAS_VALIDACION)}**, que nunca vio "
                "(validación por tiempo, como en producción). Cada fraude pesa lo que "
                f"**{dec(metricas['peso_clase_fraude'], 1)} transacciones legítimas** (class weights). "
                "Nunca se usa accuracy: con 3,5 % de fraude, un modelo que nunca alerta tendría 96,5 %.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC-AUC", dec(gbt["roc_auc"]), delta="Meta ≥ 0,85 ✓" if gbt["roc_auc"] >= 0.85 else "Meta ≥ 0,85 ✗",
              delta_color="normal" if gbt["roc_auc"] >= 0.85 else "inverse", border=True,
              help="Capacidad de ordenar fraudes por encima de legítimas (0,5 = azar, 1 = perfecto)")
    m2.metric("Recall", dec(gbt["recall"]), delta="Meta ≥ 0,70 ✓" if gbt["recall"] >= 0.70 else "Meta ≥ 0,70 ✗",
              delta_color="normal" if gbt["recall"] >= 0.70 else "inverse", border=True,
              help="De cada 100 fraudes, cuántos detecta (umbral 0,5)")
    m3.metric("Precision", dec(gbt["precision"]), border=True,
              help="De cada 100 alertas, cuántas son fraude real (umbral 0,5)")
    m4.metric("PR-AUC", dec(gbt["pr_auc"]), delta=f"{dec(gbt['pr_auc'] / (base_pr / 100), 0)} veces el azar",
              delta_color="off", delta_arrow="off", border=True,
              help=f"Área bajo la curva Precision-Recall. Al azar sería {dec(base_pr / 100)}")

    c1, c2 = st.columns(2)
    with c1:
        nombres = ["ROC-AUC", "PR-AUC", "Recall", "Precision", "F1"]
        claves = ["roc_auc", "pr_auc", "recall", "precision", "f1"]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="GBT con class weights", x=nombres, y=[gbt[k] for k in claves], marker_color=LEGITIMA,
                             text=[dec(gbt[k], 2) for k in claves], textposition="outside", textfont_color=TEXTO_2,
                             hovertemplate="%{x}: %{y:.3f}<extra>GBT</extra>"))
        if lr:
            fig.add_trace(go.Bar(name="Regresión logística (base)", x=nombres, y=[lr[k] for k in claves],
                                 marker_color=MODELO_2, text=[dec(lr[k], 2) for k in claves], textposition="outside",
                                 textfont_color=TEXTO_2, hovertemplate="%{x}: %{y:.3f}<extra>Regresión logística</extra>"))
        fig.update_layout(title="Métricas en validación: modelo final vs. línea base", barmode="group",
                          bargap=0.25, bargroupgap=0.08)
        fig.update_yaxes(range=[0, 1.1])
        grafico(estilo(fig, alto=360, leyenda=True))
    with c2:
        imp = pd.DataFrame(metricas["importancia_variables_top20"]).head(15).iloc[::-1]
        fig = go.Figure(go.Bar(x=imp["importancia"], y=imp["variable"], orientation="h", marker_color=LEGITIMA,
                               hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        fig.update_layout(title="Las 15 variables que más pesan en el modelo", bargap=0.3)
        grafico(estilo(fig, alto=360))

    fig = go.Figure()
    for etiqueta, nombre, color in [(0, "Legítimas", LEGITIMA), (1, "Fraudes", FRAUDE)]:
        sub = pred[pred["isFraud"] == etiqueta]["prob_fraude"]
        fig.add_trace(go.Histogram(x=sub, name=nombre, marker_color=color, opacity=0.75, xbins=dict(size=0.025),
                                   histnorm="percent",
                                   hovertemplate=f"{nombre}<br>Probabilidad %{{x}}<br>%{{y:.1f}} % del grupo<extra></extra>"))
    fig.update_layout(title="¿Separa el modelo los fraudes de las legítimas? Distribución del puntaje en validación",
                      barmode="overlay")
    fig.update_xaxes(title="Puntaje de fraude del modelo")
    fig.update_yaxes(title="% de transacciones del grupo")
    grafico(estilo(fig, alto=320, leyenda=True))
    st.caption("Las legítimas se acumulan con puntajes bajos y los fraudes con puntajes altos: esa separación es lo "
               "que mide el ROC-AUC. Las variables más importantes (correo, monto, tarjeta, producto) coinciden con "
               "los segmentos de riesgo del panorama. Las variables V las creó Vesta y su significado no es público.")

# ================================================================ P3
with tab3:
    costos = metricas.get("costos_supuestos", {"revision_usd": 5.0, "contracargo_usd": 25.0})
    st.markdown("Cada alerta se revisa a mano y cada fraude que se escapa cuesta su monto más un contracargo. "
                "Subir el umbral reduce alertas pero deja pasar más fraude; bajarlo hace lo contrario. "
                "**Los costos son supuestos del equipo:** cámbialos para ver cómo se mueve el umbral óptimo.")
    a, b, c = st.columns([1, 1, 2])
    revision = a.number_input("Costo de revisar una alerta (US$)", 0.0, 100.0, float(costos["revision_usd"]), 1.0,
                              format="%.0f")
    contracargo = b.number_input("Contracargo por fraude no detectado (US$)", 0.0, 500.0,
                                 float(costos["contracargo_usd"]), 5.0, format="%.0f")
    curva = curva_costo(pred["prob_fraude"].to_numpy(), pred["isFraud"].to_numpy(),
                        pred["TransactionAmt"].to_numpy(), revision, contracargo, len(DIAS_VALIDACION))
    optimo = curva.loc[curva["costo"].idxmin()]
    sin_modelo = float(pred.loc[pred["isFraud"] == 1, "TransactionAmt"].sum()) + int(pred["isFraud"].sum()) * contracargo
    umbral = c.slider("Umbral de alerta", 0.05, 0.95, float(optimo["umbral"]), 0.05, key="umbral_p3",
                      help=f"Con estos costos, el umbral de menor costo es {dec(optimo['umbral'], 2)}")
    fila = curva.iloc[(curva["umbral"] - umbral).abs().idxmin()]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Alertas por día", entero(fila["alertas_por_dia"]), border=True,
              help="Transacciones que el equipo de revisión tendría que atender cada día")
    m2.metric("Fraudes detectados", pct(100 * fila["recall"], 1),
              delta=f"{entero(fila['tp'])} de {entero(fila['tp'] + fila['fn'])}", delta_color="off", delta_arrow="off",
              border=True)
    m3.metric("Falsas alarmas", pct(100 * fila["fp"] / (fila["fp"] + fila["tn"]), 1),
              delta=f"{entero(fila['fp'])} legítimas revisadas", delta_color="off", delta_arrow="off", border=True)
    m4.metric(f"Costo en {len(DIAS_VALIDACION)} días (US$)", corto(fila["costo"]),
              delta=f"Ahorro {pct(100 * (1 - fila['costo'] / sin_modelo), 0)} vs. sin modelo", border=True,
              help=f"Sin modelo (todo fraude se pierde): {usd(sin_modelo)}")

    c1, c2 = st.columns([3, 2])
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=curva["umbral"], y=curva["costo"], mode="lines", line=dict(color=LEGITIMA, width=2.5),
                                 name="Costo con el modelo",
                                 hovertemplate="Umbral %{x:.2f}<br>Costo US$ %{y:,.0f}<extra></extra>"))
        fig.add_hline(y=sin_modelo, line=dict(color=REFERENCIA, dash="dash", width=1),
                      annotation_text=f"Sin modelo: {usd(sin_modelo)}", annotation_position="top left",
                      annotation_font_color=TEXTO_2)
        fig.add_trace(go.Scatter(x=[optimo["umbral"]], y=[optimo["costo"]], mode="markers+text", name="Óptimo",
                                 marker=dict(color=LEGITIMA, size=11, line=dict(color="white", width=2)),
                                 text=[f"Óptimo {dec(optimo['umbral'], 2)}"], textposition="bottom center",
                                 textfont_color=TEXTO, hoverinfo="skip"))
        fig.add_vline(x=umbral, line=dict(color=TEXTO, width=1))
        fig.update_layout(title="Costo total en los 30 días de validación según el umbral")
        fig.update_xaxes(title="Umbral de alerta")
        fig.update_yaxes(title="US$", rangemode="tozero")
        grafico(estilo(fig, alto=360))
    with c2:
        matriz = np.array([[fila["tn"], fila["fp"]], [fila["fn"], fila["tp"]]])
        textos = [[f"{entero(fila['tn'])}<br>legítimas aprobadas", f"{entero(fila['fp'])}<br>falsas alarmas"],
                  [f"{entero(fila['fn'])}<br>fraudes no detectados", f"{entero(fila['tp'])}<br>fraudes detectados"]]
        fig = go.Figure(go.Heatmap(z=np.log10(matriz + 1), x=["Predice legítima", "Predice fraude"],
                                   y=["Es legítima", "Es fraude"], text=textos, texttemplate="%{text}",
                                   textfont=dict(size=13), colorscale=AZULES, showscale=False, hoverinfo="skip",
                                   xgap=3, ygap=3))
        fig.update_layout(title=f"Matriz de confusión con umbral {dec(umbral, 2)}")
        fig.update_yaxes(autorange="reversed", showgrid=False)
        grafico(estilo(fig, alto=360))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curva["umbral"], y=curva["recall"], name="Recall (fraudes detectados)",
                             line=dict(color=FRAUDE, width=2.5), hovertemplate="Umbral %{x:.2f}: %{y:.3f}<extra>Recall</extra>"))
    fig.add_trace(go.Scatter(x=curva["umbral"], y=curva["precision"], name="Precision (alertas que son fraude)",
                             line=dict(color=LEGITIMA, width=2.5),
                             hovertemplate="Umbral %{x:.2f}: %{y:.3f}<extra>Precision</extra>"))
    fig.add_vline(x=umbral, line=dict(color=TEXTO, width=1))
    fig.update_layout(title="El intercambio: más fraudes detectados implica más alertas falsas")
    fig.update_xaxes(title="Umbral de alerta")
    fig.update_yaxes(range=[0, 1.05])
    grafico(estilo(fig, alto=300, leyenda=True))
    st.caption("La recomendación tiene una condición: con muchas falsas alarmas, la revisión debe ser liviana "
               "(por ejemplo, una verificación por mensaje de texto) y el equipo debe poder atender las alertas diarias. "
               "Si no hay esa capacidad, conviene subir el umbral.")

# ================================================================ Monitoreo
with tab4:
    st.markdown("**Simulación de monitoreo:** se reproducen, hora por hora, las transacciones de un día del periodo "
                "de validación como si llegaran en vivo, y el modelo marca las que superan el umbral. "
                "Para el flujo real con Kafka y Spark Structured Streaming, ver la pestaña **Streaming en vivo**.")
    a, b = st.columns([1, 2])
    dia = a.selectbox("Día de validación", DIAS_VALIDACION, index=len(DIAS_VALIDACION) - 1,
                      format_func=lambda v: f"Día {v}")
    umbral_m = b.slider("Umbral de alerta", 0.05, 0.95, float(metricas["p3_umbral_optimo"]["umbral"]), 0.05,
                        key="umbral_monitoreo", help="Por defecto, el umbral óptimo calculado por el pipeline")
    del_dia = aplicar_filtros(pred[pred["dia"] == dia]).sort_values(["hora", "TransactionID"])
    del_dia = del_dia.assign(alerta=del_dia["prob_fraude"] >= umbral_m)

    columnas = {"TransactionID": "Transacción", "hora": "Hora", "TransactionAmt": "Monto (US$)",
                "prob_fraude": "Puntaje", "ProductCD": "Producto", "card6": "Tarjeta",
                "P_email_proveedor": "Correo", "DeviceType": "Dispositivo", "real": "¿Era fraude?"}

    def tabla_alertas(df: pd.DataFrame) -> pd.DataFrame:
        t = df[df["alerta"]].assign(real=np.where(df.loc[df["alerta"], "isFraud"] == 1, "Sí", "No"))
        return t[[c for c in columnas if c in t.columns]].rename(columns=columnas)

    config_tabla = {"Puntaje": st.column_config.ProgressColumn("Puntaje", min_value=0.0, max_value=1.0, format="%.2f"),
                    "Monto (US$)": st.column_config.NumberColumn(format="%.2f"),
                    "Transacción": st.column_config.NumberColumn(format="%d")}

    def tarjetas(lugar, df: pd.DataFrame) -> None:
        alertas = df[df["alerta"]]
        cols = lugar.columns(4)
        cols[0].metric("Transacciones procesadas", entero(len(df)), border=True)
        cols[1].metric("Alertas generadas", entero(len(alertas)), border=True)
        cols[2].metric("Alertas que eran fraude", entero(int(alertas["isFraud"].sum())), border=True)
        cols[3].metric("Monto en alertas (US$)", corto(alertas["TransactionAmt"].sum()), border=True)

    simular = st.button("▶ Simular la llegada de transacciones del día", type="primary")
    zona_tarjetas, zona_progreso, zona_tabla = st.empty(), st.empty(), st.empty()
    if simular:
        for h in range(24):
            parcial = del_dia[del_dia["hora"] <= h]
            tarjetas(zona_tarjetas.container(), parcial)
            zona_progreso.progress((h + 1) / 24, text=f"Hora {h:02d}:00 · últimas alertas")
            zona_tabla.dataframe(tabla_alertas(parcial).tail(8).iloc[::-1], hide_index=True,
                                 column_config=config_tabla, width="stretch")
            time.sleep(0.35)
        zona_progreso.empty()
    tarjetas(zona_tarjetas.container(), del_dia)

    por_hora = del_dia[del_dia["alerta"]].groupby(["hora", "isFraud"]).size().unstack(fill_value=0) \
        .reindex(index=range(24), columns=[0, 1], fill_value=0)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=por_hora.index, y=por_hora[1], name="Alerta correcta (fraude real)",
                         marker_color=FRAUDE, hovertemplate="Hora %{x}: %{y} fraudes<extra></extra>"))
    fig.add_trace(go.Bar(x=por_hora.index, y=por_hora[0], name="Falsa alarma (legítima)",
                         marker_color=LEGITIMA, hovertemplate="Hora %{x}: %{y} falsas alarmas<extra></extra>"))
    fig.update_layout(title=f"Alertas por hora · día {dia}", barmode="stack", bargap=0.2)
    fig.update_xaxes(title="Hora", dtick=1)
    grafico(estilo(fig, alto=300, leyenda=True))

    tabla = tabla_alertas(del_dia).sort_values("Puntaje", ascending=False)
    zona_tabla.dataframe(tabla, hide_index=True, column_config=config_tabla, width="stretch", height=380)
    st.download_button("Descargar alertas del día (CSV)", tabla.to_csv(index=False).encode("utf-8"),
                       file_name=f"alertas_dia_{dia}.csv", mime="text/csv")

# ================================================================ Streaming real (bonus)
STREAM = DATOS / "streaming" / "alertas_stream.csv"

with tab5:
    st.markdown("**Streaming real (bonus):** un productor envía a **Kafka** las transacciones de un día de validación, "
                "hora por hora; **Spark Structured Streaming** las lee en micro-lotes de 2 segundos, las califica con el "
                "mismo modelo GBT del pipeline y guarda las alertas en Gold. Este panel se actualiza solo cada 2 segundos.")
    st.caption("Productor → Kafka (tópico transacciones) → Spark Structured Streaming → modelo GBT → "
               "gold/alertas_streaming (Delta) → este panel")

    @st.fragment(run_every=2)
    def panel_streaming() -> None:
        try:
            df = pd.read_csv(STREAM).dropna(subset=["TransactionID"])
        except (FileNotFoundError, pd.errors.EmptyDataError):
            st.info("Esperando transacciones del streaming. Con Kafka arriba, lanza el consumidor y luego el productor "
                    "(README, sección Streaming).")
            return
        alertas = df[df["alerta"].astype(str) == "True"]
        c = st.columns(4)
        c[0].metric("Transacciones calificadas", entero(len(df)), border=True)
        c[1].metric("Alertas generadas", entero(len(alertas)), border=True)
        c[2].metric("Alertas que eran fraude", entero(int(alertas["isFraud"].sum())), border=True)
        c[3].metric("Latencia media", dec(df["latencia_s"].mean(), 1) + " s", border=True,
                    help="Tiempo entre el envío a Kafka y la calificación del modelo")
        dia_actual = int(df["dia"].iloc[-1])
        dias = ", ".join(str(int(d)) for d in df["dia"].drop_duplicates())
        st.caption(f"Días recibidos en esta corrida: {dias}. El gráfico muestra el día en curso.")
        por_hora = alertas[alertas["dia"] == dia_actual].groupby(["hora", "isFraud"]).size().unstack(fill_value=0) \
            .reindex(index=range(24), columns=[0, 1], fill_value=0)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=por_hora.index, y=por_hora[1], name="Alerta correcta (fraude real)", marker_color=FRAUDE,
                             hovertemplate="Hora %{x}: %{y} fraudes<extra></extra>"))
        fig.add_trace(go.Bar(x=por_hora.index, y=por_hora[0], name="Falsa alarma (legítima)", marker_color=LEGITIMA,
                             hovertemplate="Hora %{x}: %{y} falsas alarmas<extra></extra>"))
        fig.update_layout(title=f"Alertas por hora a medida que llegan · día {dia_actual}",
                          barmode="stack", bargap=0.2)
        fig.update_xaxes(title="Hora", dtick=1, range=[-0.5, 23.5])
        grafico(estilo(fig, alto=300, leyenda=True))
        ultimas = alertas.sort_values(["calificado_ts", "hora"], ascending=False).head(10).assign(
            real=lambda d: np.where(d["isFraud"] == 1, "Sí", "No"),
            calificado_ts=lambda d: pd.to_datetime(d["calificado_ts"]).dt.strftime("%H:%M:%S"))
        cols = {"calificado_ts": "Calificada (UTC)", "TransactionID": "Transacción", "hora": "Hora",
                "TransactionAmt": "Monto (US$)", "prob_fraude": "Puntaje", "ProductCD": "Producto",
                "card6": "Tarjeta", "P_email_proveedor": "Correo", "real": "¿Era fraude?"}
        st.markdown("**Últimas alertas**")
        st.dataframe(ultimas[[k for k in cols if k in ultimas.columns]].rename(columns=cols), hide_index=True,
                     width="stretch",
                     column_config={"Puntaje": st.column_config.ProgressColumn("Puntaje", min_value=0.0, max_value=1.0,
                                                                               format="%.2f"),
                                    "Transacción": st.column_config.NumberColumn(format="%d"),
                                    "Monto (US$)": st.column_config.NumberColumn(format="%.2f")})

    panel_streaming()

st.divider()
st.caption("Pipeline: CSV de Kaggle → Bronze → Silver → Gold en Spark 3.5 y Delta Lake · modelo GBT de Spark MLlib "
           "registrado en MLflow · dashboard en Streamlit sobre las tablas Gold. "
           f"Equipo: Julian Rodriguez, Samuel Serrano y Samantha De Aguas · [código en GitHub]({REPO})")
