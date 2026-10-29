"""Figuras del artículo a partir de los datos de data/.

    figures/fig_flujo.pdf       Phi*(Theta) para n impar (data/phistar.csv)
    figures/fig_asintotica.pdf  coeficiente de 1/n^2 y anchura de la ventana
                                (data/edges_ed.csv y data/bethe_large.csv)

Colores: paleta categórica validada (orden fijo; azul, naranja, aguamarina,
amarillo). Tipografía Computer Modern y coma decimal, como el texto.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from holonomy import asymptotics  # noqa: E402
from holonomy.report import FIGURES, read_csv  # noqa: E402

SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
MARKERS = ("o", "s", "^", "D")
INK = "#3d3d3a"
MUTED = "#8a8980"
GRID = "#e6e5df"
TEXTWIDTH_IN = 6.3  # A4 con márgenes de 2,5 cm

plt.rcParams.update({
    # Computer Modern de Matplotlib (sin depender de una instalación de LaTeX)
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "font.size": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.4,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def comma(x, _pos=None):
    """Formato de marca con coma decimal."""
    return f"${x:g}$".replace(".", "{,}")


def fig_flujo():
    rows = read_csv("phistar.csv")
    fig, ax = plt.subplots(figsize=(0.75 * TEXTWIDTH_IN, 2.5))
    for k, n in enumerate((7, 9, 11, 13)):
        pts = [(r["theta_over_pi"], r["phistar_over_pi"]) for r in rows if int(r["n"]) == n]
        x, y = map(np.array, zip(*pts))
        ax.plot(x, y, color=SERIES[k], marker=MARKERS[k], markevery=10, markersize=4,
                label=f"$n={n}$")
        # etiqueta directa bajo el punto de despegue (borde de la ventana)
        i = int(np.argmax(y > 1e-6)) - 1
        ax.annotate(f"${n}$", (x[i], 0), xytext=(0, -9), textcoords="offset points",
                    ha="center", va="center", fontsize=8, color=INK)
    ax.set_xlim(0.95, 1.0)
    ax.set_ylim(-0.12, 1.02)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel(r"$\Theta/\pi$")
    ax.set_ylabel(r"$\Phi_\star/\pi$")
    ax.xaxis.set_major_formatter(FuncFormatter(comma))
    ax.yaxis.set_major_formatter(FuncFormatter(comma))
    ax.legend(frameon=False, loc="upper left")
    out = FIGURES / "fig_flujo.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"escrito {out}")


def fig_asintotica():
    rows = [r for r in read_csv("edges_ed.csv") if r["n"] >= 6] + read_csv("bethe_large.csv")
    rows.sort(key=lambda r: r["n"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(0.95 * TEXTWIDTH_IN, 2.5))
    for p, name in ((0, "$n$ par"), (1, "$n$ impar")):
        sel = [r for r in rows if int(r["n"]) % 2 == p]
        n = np.array([r["n"] for r in sel])
        ax1.plot(n, np.abs([r["coef"] for r in sel]), color=SERIES[p], marker=MARKERS[p],
                 markersize=3.5, label=name)
        ax2.plot(n, [r["psi_over_pi"] for r in sel], color=SERIES[p], marker=MARKERS[p],
                 markersize=3.5, label=name)
    ax1.axhline(np.pi ** 2 / 6, color=MUTED, ls="--", lw=1.0, label=r"$\pi^2/6$")
    ax2.axhline(asymptotics.psi_infinity_over_pi(), color=MUTED, ls="--", lw=1.0,
                label=r"$\psi_\infty/\pi$")
    ax1.set_ylabel(r"$n^2\,|v(n,0)-v(n,\pi)|$")
    ax2.set_ylabel(r"$\psi(n)/\pi$")
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_xlabel(r"$n$")
        ax.yaxis.set_major_formatter(FuncFormatter(comma))
        ax.legend(frameon=False)
    ax2.set_ylim(0, None)
    fig.tight_layout(w_pad=2.0)
    out = FIGURES / "fig_asintotica.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"escrito {out}")


def main():
    FIGURES.mkdir(exist_ok=True)
    fig_flujo()
    fig_asintotica()


if __name__ == "__main__":
    main()
