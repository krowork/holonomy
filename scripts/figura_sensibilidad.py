"""Figura de robustez_mps.tex: sensibilidad a la holonomía frente a n.

Panel (a), n <= 18, semilogarítmico (una recta es decaimiento exponencial):
  - AKLT: |p(n, 0) - p(n, pi)| del SDP exacto (tabla de la sección AKLT, aklt_sdp_ti.py);
  - Néel, kappa = 0,4, n par e impar, y datos de Werner (kappa = 1, testigo crítico):
    |v(n, 0) - v(n, pi)| de data/neel_u1.csv.
Panel (b), Bethe ansatz hasta n = 3001, doble logarítmico (una recta es una potencia):
  kappa = 0,4 y 0,8 por paridad y kappa = 1, de data/neel_bethe.csv, con las
  pendientes de referencia n^-2 y n^-3.

Color = kappa (o AKLT), igual en los dos paneles; trazo = paridad (continuo n par,
discontinuo n impar; punteado el crítico, con las dos paridades juntas). Paleta
validada (todos los pares, modo claro): visión normal >= 20; daltonismo 6,9, por
eso la paridad va también en el trazo y las curvas llevan etiqueta directa.

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

BLUE, RED, AQUA, YELLOW = "#2a78d6", "#e34948", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#1f1f1e", "#5f5e58", "#e4e3de"
COLOR = {0.4: RED, 0.8: AQUA, 1.0: YELLOW}


def neel(kappa, parity):
    out = []
    with open(ROOT / "data" / "neel_u1.csv") as f:
        for r in csv.DictReader(f):
            n = int(r["n"])
            if float(r["kappa"]) == kappa and n >= 8 and n % 2 == parity:
                out.append((n, abs(float(r["v0"]) - float(r["vpi"]))))
    return sorted(out)


def bethe(kappa, paridad):
    out = []
    with open(ROOT / "data" / "neel_bethe.csv") as f:
        for r in csv.DictReader(f):
            if (float(r["kappa"]) == kappa and r["paridad"] == paridad and r["frontera"] == "0"
                    and float(r["s"]) > 1e-12):
                out.append((int(r["n"]), float(r["s"])))
    return sorted(out)


def estilo(ax):
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "text.color": INK})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw={"width_ratios": [1, 1.15]})

    # --- (a) n <= 18 ---------------------------------------------------------
    series = [
        ("AKLT", sorted((n, abs(a - b)) for n, (a, b) in AKLT.items()), BLUE, "o", "-"),
        (r"$\kappa=0{,}4$, $n$ par", neel(0.4, 0), RED, "s", "-"),
        (r"$\kappa=0{,}4$, $n$ impar", neel(0.4, 1), RED, "^", "--"),
        (r"$\kappa=1$ (crítico)", sorted(neel(1.0, 0) + neel(1.0, 1)), YELLOW, "D", ":"),
    ]
    for label, pts, color, marker, ls in series:
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=color, lw=2, ls=ls, marker=marker, ms=5,
                mec="white", mew=0.8, label=label)
        print(f"(a) {label}: " + ", ".join(f"{n}:{y:.2e}" for n, y in pts))
    ax.set_yscale("log")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$|p(n,0)-p(n,\pi)|$")
    ax.set_xticks(range(4, 19, 2))
    ax.set_title(r"(a) $n\leq18$: SDP y diagonalización exacta", fontsize=9, loc="left", color=INK)
    estilo(ax)
    ax.legend(frameon=False, loc="lower right", fontsize=7.5)

    # --- (b) Bethe, n <= 3001 ------------------------------------------------
    curvas = [
        (0.4, "par", "-", r"$\kappa=0{,}4$, par: $e^{-n/5{,}1}$"),
        (0.4, "impar", "--", r"$\kappa=0{,}4$, impar: $n^{-3}$"),
        (0.8, "par", "-", r"$\kappa=0{,}8$, par: $e^{-n/74}$"),
        (0.8, "impar", "--", r"$\kappa=0{,}8$, impar: $n^{-3}$"),
        (1.0, "par", ":", r"$\kappa=1$: $n^{-2}$"),
    ]
    for kappa, par, ls, label in curvas:
        xs, ys = zip(*bethe(kappa, par))
        bx.plot(xs, ys, color=COLOR[kappa], lw=2, ls=ls, label=label)
        print(f"(b) {label}: n = {xs[0]}..{xs[-1]}, s = {ys[0]:.2e}..{ys[-1]:.2e}")
    # pendientes de referencia
    for p, y0, txt in ((2, 8e-5, r"$\propto n^{-2}$"), (3, 1.5e-6, r"$\propto n^{-3}$")):
        xr = [300, 3000]
        yr = [y0 * (x / 300) ** -p for x in xr]
        bx.plot(xr, yr, color=MUTED, lw=0.8, ls=(0, (1, 2)))
        bx.text(3200, yr[1], txt, color=MUTED, fontsize=7.5, va="center")
    bx.set_xscale("log")
    bx.set_yscale("log")
    bx.set_xlim(6, 6000)
    bx.set_ylim(1e-12, 1e-1)
    bx.set_xlabel(r"$n$")
    bx.set_ylabel(r"$|v(n,0)-v(n,\pi)|$")
    bx.set_title(r"(b) Bethe ansatz, $n\leq3001$", fontsize=9, loc="left", color=INK)
    estilo(bx)
    bx.legend(frameon=False, loc="lower left", fontsize=7.5)

    fig.tight_layout()
    out = ROOT / "docs" / "sensibilidad.pdf"
    fig.savefig(out)
    print("escrito", out)


if __name__ == "__main__":
    main()
