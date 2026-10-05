"""Estilo común de los gráficos de los notebooks y guardado en docs/evidencias/.

Paleta validada para daltonismo y contraste (azul = legítima, rojo = fraude).
"""
import matplotlib.pyplot as plt

from src.config import RAIZ

LEGITIMA = "#2a78d6"
FRAUDE = "#e34948"
MODELO_2 = "#eb6834"      # segundo color categórico (comparar dos modelos)
TEXTO = "#0b0b0b"
TEXTO_2 = "#52514e"
REJILLA = "#e4e3dd"
SUPERFICIE = "#fcfcfb"
REFERENCIA = "#52514e"    # líneas de referencia (promedio global, umbral)

EVIDENCIAS = RAIZ / "docs" / "evidencias"


def estilo() -> None:
    plt.rcParams.update({
        "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE, "savefig.facecolor": SUPERFICIE,
        "axes.edgecolor": REJILLA, "axes.labelcolor": TEXTO_2, "axes.titlecolor": TEXTO,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True, "grid.color": REJILLA, "grid.linewidth": 0.8,
        "xtick.color": TEXTO_2, "ytick.color": TEXTO_2, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.frameon": False, "legend.fontsize": 10, "lines.linewidth": 2,
        "figure.dpi": 100, "font.size": 10,
    })


def guardar(fig, nombre: str) -> None:
    """Guarda el gráfico como evidencia (PNG) en docs/evidencias/."""
    EVIDENCIAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(EVIDENCIAS / f"{nombre}.png", dpi=150, bbox_inches="tight")


def etiquetar_barras(ax, barras, formato="{:,.0f}", horizontal=False) -> None:
    """Escribe el valor al final de cada barra, en color de texto (nunca el color de la serie)."""
    for b in barras:
        if horizontal:
            v = b.get_width()
            ax.annotate(formato.format(v), (v, b.get_y() + b.get_height() / 2), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=9, color=TEXTO)
        else:
            v = b.get_height()
            ax.annotate(formato.format(v), (b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=9, color=TEXTO)
