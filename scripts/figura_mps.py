"""Figura de robustez_mps.tex: déficit y sensibilidad de los MPS con U(1) y D = 2, frente a lambda^(2n).

Datos: data/mps_su2_familia.csv (cuatro puntos (r, t) de la familia real, AKLT incluido) y
data/mps_su2_canon.csv (forma real de tres MPS aleatorios), de scripts/mps_su2.py; los mismos
siete casos de la tabla del documento. Como allí, se excluye un n entero si algún Theta de la
malla no está certificado a 1e-8 (p_sup - p_inf), y s = max - min de p_sup sobre la malla.

Panel (a): (1 - p(n, 0)) / lambda^(2n), semilogarítmico; plano si el déficit va como lambda^(2n).
Panel (b): s / lambda^(2n); plano si lambda viene del sector 0 (mu_0 >= mu_off), y cae
geométricamente si lo da el sector de carga +-1 (mu_off > mu_0).

Color = qué sector da lambda (igual en los dos paneles); marcador = caso, con etiqueta en la
leyenda. Misma paleta y estilo que figura_sensibilidad.py.

Escribe docs/mps_lambda.pdf.
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

BLUE, RED = "#2a78d6", "#e34948"
INK, MUTED, GRID = "#1f1f1e", "#5f5e58", "#e4e3de"
CERT = 1e-8

# (fichero, caso, etiqueta, sector que da lambda, marcador); el sector, de la tabla del documento
CASOS = [
    ("familia", "r1_t2", r"$(1;\,2)$, AKLT", 0, "o"),
    ("familia", "r0.7_t1.5", r"$(0{,}7;\,1{,}5)$", 0, "s"),
    ("familia", "r1_t1.5", r"$(1;\,1{,}5)$", 1, "o"),
    ("familia", "r1.25_t2", r"$(1{,}25;\,2)$", 1, "s"),
    ("canon", "aleatorio1_canon", "aleatorio 1", 1, "^"),
    ("canon", "aleatorio6_canon", "aleatorio 6", 1, "D"),
    ("canon", "aleatorio7_canon", "aleatorio 7", 1, "v"),
]


def leer(modo):
    filas = defaultdict(lambda: defaultdict(list))
    with open(ROOT / "data" / f"mps_su2_{modo}.csv") as f:
        for r in csv.DictReader(f):
            filas[r["caso"]][int(r["n"])].append(
                (float(r["Theta"]), float(r["p_sup"]), float(r["p_inf"]), float(r["lam"])))
    return filas


def serie(filas):
    """[(n, (1 - p(n,0))/lam^2n, s/lam^2n)] con los n certificados."""
    out = []
    for n, pts in sorted(filas.items()):
        if max(a - b for _, a, b, _ in pts) > CERT:
            continue
        lam = pts[0][3]
        p0 = next(a for t, a, _, _ in pts if t == 0.0)
        ps = [a for _, a, _, _ in pts]
        out.append((n, (1 - p0) / lam ** (2 * n), (max(ps) - min(ps)) / lam ** (2 * n)))
    return out


def estilo(ax):
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "text.color": INK})
    datos = {m: leer(m) for m in ("familia", "canon")}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 3.9))

    for modo, caso, label, sector, marker in CASOS:
        pts = serie(datos[modo][caso])
        ns, defs, ss = zip(*pts)
        color = BLUE if sector == 0 else RED
        kw = dict(color=color, lw=1.4, marker=marker, ms=5, mec="white", mew=0.8)
        ax.plot(ns, defs, **kw, label=label)
        bx.plot(ns, ss, **kw)
        print(f"{label}: n = {list(ns)}; déficit/l^2n = {min(defs):.0f}..{max(defs):.0f}; "
              f"s/l^2n último = {ss[-1]:.2g}")

    # guía: caída geométrica de s/lambda^2n a 0,57 por sitio (centro de 0,52-0,62)
    xr = [8, 12]
    yr = [7 * 0.57 ** (x - 8) for x in xr]
    bx.plot(xr, yr, color=MUTED, lw=0.8, ls=(0, (1, 2)))
    bx.text(12.15, yr[1], r"$\propto0{,}57^{\,n}$", color=MUTED, fontsize=7.5, va="center")

    for a in (ax, bx):
        a.set_yscale("log")
        a.set_xlabel(r"$n$")
        a.set_xticks(range(6, 13))
        a.set_xlim(5.6, 13.2)
        estilo(a)
    ax.set_ylabel(r"$[1-p(n,0)]/\lambda^{2n}$")
    ax.set_title(r"(a) déficit", fontsize=9, loc="left", color=INK)
    bx.set_ylabel(r"$s/\lambda^{2n}$")
    bx.set_title(r"(b) sensibilidad", fontsize=9, loc="left", color=INK)
    ax.set_ylim(5, 600)
    bx.set_ylim(0.5, 600)

    h, lab = ax.get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, loc="lower center", fontsize=7.5, ncol=4,
               bbox_to_anchor=(0.5, 0.0))
    fig.text(0.5, 0.115, r"azul: $\lambda=\mu_0$ (sector $0$);  rojo: $\lambda=\mu_{\rm off}>\mu_0$"
             r" (sector $\pm1$)", ha="center", fontsize=7.5, color=MUTED)

    fig.tight_layout(rect=(0, 0.14, 1, 1))
    out = ROOT / "docs" / "mps_lambda.pdf"
    fig.savefig(out)
    print("escrito", out)


if __name__ == "__main__":
    main()
