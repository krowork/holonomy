"""Figura de robustez_mps.tex: sensibilidad a la holonomía frente a n.

Tres casos en la misma escala semilogarítmica (una recta es decaimiento exponencial):
  - AKLT: |p(n, 0) - p(n, pi)| del SDP exacto (tabla de la sección AKLT, aklt_sdp_ti.py);
  - Néel, kappa = 0,4, n par e impar: |v(n, 0) - v(n, pi)| de data/neel_u1.csv;
  - datos de Werner (kappa = 1, testigo crítico): ídem.

Escribe docs/sensibilidad.pdf.
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# p(n, 0) y p(n, pi) de AKLT, como en la tabla del documento
AKLT = {4: (0.975610, 0.902726), 5: (0.97036, 0.997422), 6: (0.9997978, 0.9993839),
        7: (0.9999421, 0.9999770), 8: (0.99999658, 0.99999624)}

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#1f1f1e", "#5f5e58", "#e4e3de"


def neel(kappa, parity):
    out = []
    with open(ROOT / "data" / "neel_u1.csv") as f:
        for r in csv.DictReader(f):
            n = int(r["n"])
            if float(r["kappa"]) == kappa and n >= 8 and n % 2 == parity:
                out.append((n, abs(float(r["v0"]) - float(r["vpi"]))))
    return sorted(out)


def main():
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "text.color": INK})
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    series = [
        ("AKLT", sorted((n, abs(a - b)) for n, (a, b) in AKLT.items()), BLUE, "o", "-"),
        (r"Néel $\kappa=0{,}4$, $n$ par", neel(0.4, 0), ORANGE, "s", "-"),
        (r"Néel $\kappa=0{,}4$, $n$ impar", neel(0.4, 1), AQUA, "^", "--"),
        (r"Werner $\kappa=1$ (crítico)", sorted(neel(1.0, 0) + neel(1.0, 1)), YELLOW, "D", ":"),
    ]
    for label, pts, color, marker, ls in series:
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=color, lw=2, ls=ls, marker=marker, ms=5,
                mec="white", mew=0.8, label=label)
        print(f"{label}: " + ", ".join(f"{n}:{y:.2e}" for n, y in pts))
    ax.set_yscale("log")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$|p(n,0)-p(n,\pi)|$")
    ax.set_xticks(range(4, 19, 2))
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    out = ROOT / "docs" / "sensibilidad.pdf"
    fig.savefig(out)
    print("escrito", out)


if __name__ == "__main__":
    main()
