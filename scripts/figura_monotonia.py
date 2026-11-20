"""Figura de robustez_mps.tex: forma de v(n, Theta) en Theta, normalizada, en el anillo de Néel.

Datos: data/monotonia_theta.csv (scripts/monotonia_theta.py: dual completo en (Delta, Phi) y
Bethe, 33 valores de Theta en [0, pi]). Se dibuja
    u(Theta) = [v(n, Theta) - v(n, pi)] / [v(n, 0) - v(n, pi)],
que vale 1 en Theta = 0 y 0 en Theta = pi con las dos paridades (con n impar el numerador y el
denominador son negativos). Una curva monótona en u es una v monótona. Se omiten los (kappa, n)
con |v(n, 0) - v(n, pi)| < 1e-9, donde la forma es redondeo.

Referencias (grises):
  - cos(Theta/2): n par con gap, v ~ v(n,0) - s (1 - cos Theta/2), el doblete túnel;
  - ((pi - Theta)/pi)^2: parábola con vértice en Theta = pi (n par crítico; n impar con gap
    tiende a ella, la banda del espinón);
  - 1 - (Theta/pi)^2: parábola con vértice en Theta = 0 (n impar crítico).

Panel (a) n par, (b) n impar. Color = kappa, como en figura_sensibilidad.py; trazo fino y claro
el n menor, grueso el mayor resoluble.

Escribe docs/forma_theta.pdf.
"""
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

RED, AQUA, YELLOW = "#e34948", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#1f1f1e", "#5f5e58", "#e4e3de"
COLOR = {0.4: RED, 0.8: AQUA, 1.0: YELLOW}
KTXT = {0.4: "0{,}4", 0.8: "0{,}8", 1.0: "1"}
TOL = 1e-9


def leer():
    d = defaultdict(list)
    with open(ROOT / "data" / "monotonia_theta.csv") as f:
        for r in csv.DictReader(f):
            d[(float(r["kappa"]), int(r["n"]))].append((float(r["Theta"]), float(r["v"])))
    return {k: sorted(v) for k, v in d.items()}


def forma(pts):
    t, v = zip(*pts)
    s = v[0] - v[-1]
    if abs(s) < TOL:
        return None
    return [x / math.pi for x in t], [(y - v[-1]) / s for y in v]


def estilo(ax):
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "text.color": INK})
    d = leer()
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)

    x = [i / 200 for i in range(201)]
    ref = {
        "cos": ([math.cos(math.pi * t / 2) for t in x], r"$\cos\frac{\Theta}{2}$"),
        "pi": ([(1 - t) ** 2 for t in x], r"$(1-\Theta/\pi)^2$"),
        "cero": ([1 - t ** 2 for t in x], r"$1-(\Theta/\pi)^2$"),
    }
    # n pequeño y mayor n resoluble, por paridad y kappa
    casos = {0: {0.4: (8, 64), 0.8: (8, 128), 1.0: (8, 128)},
             1: {0.4: (9, 129), 0.8: (9, 129), 1.0: (9, 129)}}
    refs_panel = {0: ("cos", "pi"), 1: ("pi", "cero")}

    for par, ax in zip((0, 1), axs):
        for key in refs_panel[par]:
            y, txt = ref[key]
            ax.plot(x, y, color=MUTED, lw=0.9, ls=(0, (1, 2)), zorder=3)
        for kappa, (n1, n2) in casos[par].items():
            for n, lw, alpha in ((n1, 1.0, 0.45), (n2, 2.0, 1.0)):
                f = forma(d[(kappa, n)])
                if f is None:
                    continue
                lab = rf"$\kappa={KTXT[kappa]}$, $n={n}$"
                ax.plot(*f, color=COLOR[kappa], lw=lw, alpha=alpha, label=lab, zorder=2)
                print(f"{'par' if par == 0 else 'impar'} {lab}: u(pi/2) = {f[1][16]:.3f}")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.02, 1.04)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])
        ax.set_xlabel(r"$\Theta$")
        estilo(ax)
        ax.legend(frameon=False, loc="lower left", fontsize=7)

    a, b = axs
    a.set_ylabel(r"$\dfrac{v(n,\Theta)-v(n,\pi)}{v(n,0)-v(n,\pi)}$")
    a.set_title(r"(a) $n$ par", fontsize=9, loc="left", color=INK)
    b.set_title(r"(b) $n$ impar", fontsize=9, loc="left", color=INK)
    # etiquetas directas de las referencias
    a.text(0.80, 0.42, ref["cos"][1], color=MUTED, fontsize=8)
    a.text(0.43, 0.19, ref["pi"][1], color=MUTED, fontsize=7.5)
    b.text(0.80, 0.52, ref["cero"][1], color=MUTED, fontsize=7.5)
    b.text(0.47, 0.07, ref["pi"][1], color=MUTED, fontsize=7.5)

    fig.tight_layout()
    out = ROOT / "docs" / "forma_theta.pdf"
    fig.savefig(out)
    print("escrito", out)


if __name__ == "__main__":
    main()
