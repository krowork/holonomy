"""Figuras de sensibilidad_celda3.tex (docs/figs/*.pdf).

  fig_rangos.pdf    rangos de Kantorovich, de (F_x) y ventanas de sensibilidad con datos sigma_N / sigma_inf
                    (valores de kantorovich_datos.py, optimiza_parametros.py y certifica_Fx.py; Sección sec:datos_ext);
  fig_deficit.pdf   cota del déficit delta_n frente a la mitad del término principal, n = 12, 15 y 18: el cruce es eps_n^*;
  fig_radios.pdf    radios por n: eps_n del Teorema principal, eps_n^*, eps_F(n), lo que sugiere la ED y el radio
                    de convergencia de lambda_0.

Uso:
  python scripts/figuras.py                    dibuja las tres (n = 12, 15 del déficit desde docs/figs/deficit_n*.csv)
  python scripts/figuras.py deficit sr12q.npz  recalcula docs/figs/deficit_n12.csv con la serie cuártica de n = 12
                                               (la de serie_respuesta.py 12 30 sr12q.npz 8 cubica cuarta; ~2 min);
                                               igual con sr15q.npz (serie_respuesta.py 15 30 sr15q.npz 6 cubica cuarta k0)
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

AQUI = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(AQUI, '..', 'docs', 'figs')
CSV = lambda n: os.path.join(FIGS, f'deficit_n{n}.csv')

# paleta categórica (orden fijo) y tintas
AZUL, NARANJA, AQUA, AMARILLO = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
TINTA, TINTA2, REJILLA = '#0b0b0b', '#52514e', '#e4e3df'

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 9, 'axes.labelsize': 9, 'legend.fontsize': 8,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.edgecolor': TINTA2, 'axes.labelcolor': TINTA,
    'xtick.color': TINTA2, 'ytick.color': TINTA2, 'axes.linewidth': 0.6, 'lines.linewidth': 1.4,
    'legend.frameon': False, 'mathtext.fontset': 'cm', 'pdf.fonttype': 42,
})


def rejilla(ax, eje='both'):
    ax.grid(True, which='major', axis=eje, color=REJILLA, linewidth=0.5)
    ax.set_axisbelow(True)
    for lado in ('top', 'right'):
        ax.spines[lado].set_visible(False)


# ------------------------------------------------------------------ 1. rangos con datos sigma_N / sigma_inf
FILAS = [  # etiqueta, Kantorovich hasta, (F_x) hasta, ventana de sensibilidad (o None), extremo inferior por desbordamiento
    (r'$n=9$, $\sigma_{12}$', 7.83e-5, 7.83e-5, (9.2e-11, 6.0e-7), False),
    (r'$n=9$, $\sigma_\infty$', 3.33e-8, 3.33e-8, None, False),
    (r'$n=12$, $\sigma_{15}$ (A)', 4.19e-3, 2.42e-3, (2.3e-23, 9.6e-4), True),
    (r'$n=12$, $\sigma_{15}$ (B)', 2.57e-3, 2.42e-3, (2.4e-23, 1.28e-3), True),
    (r'$n=12$, $\sigma_\infty$', 3.11e-8, 3.11e-8, (9.2e-24, 2.76e-15), True),
]


def fig_rangos():
    fig, ax = plt.subplots(figsize=(6.0, 2.9))
    e0 = 1e-30
    alto = 0.22
    for i, (et, ek, ef, ven, desb) in enumerate(FILAS):
        y = len(FILAS) - 1 - i
        ax.barh(y + 0.27, ek - e0, left=e0, height=alto, color='#b7d3f6', edgecolor='none')
        ax.barh(y, ef - e0, left=e0, height=alto, color=AZUL, edgecolor='none')
        if ven:
            ax.barh(y - 0.27, ven[1] - ven[0], left=ven[0], height=alto, color=NARANJA, edgecolor='none')
            if desb:
                ax.plot([ven[0]] * 2, [y - 0.27 - alto / 2, y - 0.27 + alto / 2], color=TINTA, lw=0.8, ls=':')
        else:
            ax.text(1e-29, y - 0.27, 'sin ventana', va='center', fontsize=7, color=TINTA2)
    ax.set_xscale('log')
    ax.set_xlim(e0, 3e-2)
    ax.set_yticks(range(len(FILAS)))
    ax.set_yticklabels([f[0] for f in FILAS][::-1])
    ax.set_xlabel(r'$\varepsilon$')
    ax.set_xticks([1e-30, 1e-25, 1e-20, 1e-15, 1e-10, 1e-5, 1e-2])
    rejilla(ax, 'x')
    ax.tick_params(axis='y', length=0)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    ax.legend(handles=[Patch(color='#b7d3f6', label='Kantorovich'), Patch(color=AZUL, label=r'(F$_x$) certificada'),
                       Patch(color=NARANJA, label=r'sensibilidad: $|\Delta\delta|<$ principal$/2$'),
                       Line2D([], [], color=TINTA, lw=0.8, ls=':', label='corte por desbordamiento')],
              loc='lower center', bbox_to_anchor=(0.4, 1.0), ncol=2, handlelength=1.2, columnspacing=1.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'fig_rangos.pdf'))


# ------------------------------------------------------------------ 2. déficit frente al principal
PAR = {12: (0.0262074, 2.14742e-05, 0.000766691, 0.0522126, 0.401969, 0.399606),   # optimiza_principal.py
       15: (0.0208816, 6.71838e-05, 0.000221054, 0.0564851, 0.998908, 0.99904)}
DESDE = {12: 1e-4, 15: 1e-6}


def principal(n, b4, e):
    return b4 * e ** n / (n * (2 / 3 + (8 / 3) * e ** 2))


def deficit_serie(npz):
    sys.path.insert(0, AQUI)
    _a, sys.argv = sys.argv, sys.argv[:1]
    import kantorovich_series as KS
    sys.argv = _a
    S = KS.carga(npz)
    P = KS.preparar(S)
    n = S['n']
    par = PAR[n]
    filas = []
    # los mismos intervalos [1e-30 * 1,01^k, 1e-30 * 1,01^(k+1)] del recorrido fino de optimiza_principal.py
    k = np.arange(np.ceil(np.log(DESDE[n] / 1e-30) / np.log(1.01)), 7000)
    for e in 1e-30 * 1.01 ** k:
        c = KS.cotas(S, P, e, e * 1.01, par[0], par[1:4], np.array([1, par[4], par[5]]))
        if c is None or c['h'] > 0.5:
            break
        filas.append((e, c['deficit']))
    np.savetxt(CSV(n), np.array(filas), delimiter=',', header=f'eps,cota_deficit (n={n}, PAR[{n}], intervalo [eps,1.01eps])')


def deficit_n18():
    """La misma cuenta que z23_kantorovich.analiza para n = 18 (eps_a = 1,57e-2, t' = 1,11e-2)."""
    n, kap = 18, 0.002674 / 2
    s2 = 2 ** 0.5
    best = None
    for ea in np.linspace(0.05, 0.95, 19) / (2 * s2 * n):     # el mismo óptimo que z23_kantorovich.py
        tp = (1 - 1e-9) / n - 2 * s2 * ea
        r = _n18(n, kap, ea, tp)
        if r and (best is None or r[-1][0] > best[-1][0]):
            best = r
    return np.array(best)


@np.errstate(over='ignore')
def _n18(n, kap, ea, tp, TAUH=0.6):
    Ae = 2 * 2 ** 0.5 * ea + 2 * tp
    L = 216 * Ae / tp ** 3
    ok = []
    for e in np.geomspace(1e-8, ea * 0.5, 6000):
        if e >= 4.0e-5:
            break
        r = e / ea
        D = (2 * Ae / tp) * r ** (n - 2) / (1 - r)
        kk = kap * e ** 4 - 2 * (2 * Ae / tp ** 2) * r ** (n - 4) / (1 - r)
        if kk <= 0:
            continue
        beta = 1 / (2 * kk)
        pE1 = 2 * Ae * r ** n / (1 - r) / TAUH
        d = D + pE1
        for _ in range(3):
            x = 2 * beta * d
            defE = 0.5 * x * d + 0.6 * L * x ** 3
            d = D + pE1 + defE / TAUH
        x = 2 * beta * d
        if beta * L * beta * d <= 0.5 and x <= tp / 2:
            ok.append((e, defE / TAUH))
    return ok


def fig_deficit():
    d12 = np.loadtxt(CSV(12), delimiter=',')
    d15 = np.loadtxt(CSV(15), delimiter=',')
    d18 = deficit_n18()
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    for dat, n, b4, col, emax, est in ((d12, 12, 1001.08, AZUL, 6.43e-3, 6.30e-3),
                                       (d15, 15, 4476.36, AQUA, 4.86e-3, 3.40e-3),
                                       (d18, 18, 20789.16, NARANJA, 5.0e-6, 1.1e-6)):
        e = dat[:, 0]
        ax.loglog(e, dat[:, 1], color=col, label=fr'$n={n}$: cota de $\delta_n$')
        ax.loglog(e, principal(n, b4, e) / 2, color=col, ls='--', label=fr'$n={n}$: principal$/2$')
        ax.axvline(est, color=col, lw=0.6, ls=':')
        ax.text(est, 1e-150, fr'$\varepsilon^*_{{{n}}}$', color=TINTA, fontsize=8,
                ha='right', va='bottom', rotation=90)
    ax.set_xlabel(r'$\varepsilon$')
    ax.set_ylabel('cota / término')
    ax.set_xlim(1e-8, 1e-2)
    ax.set_ylim(1e-160, 1e-15)
    ax.set_yticks([10.0 ** k for k in range(-160, -14, 20)])
    rejilla(ax)
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, 0.83), ncol=1, fontsize=7.5)
    ax.text(1.3e-8, 1e-28, r'$\delta_{18}\propto\varepsilon^{28}$, principal $\propto\varepsilon^{18}$:'
            '\n' r'el cociente cae como $\varepsilon^{10}$', fontsize=7.5, color=TINTA2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'fig_deficit.pdf'))


# ------------------------------------------------------------------ 3. radios por n
def fig_radios():
    n = np.array([6, 9, 12, 15, 18])
    eps_n = np.array([1.07e-4, 1.21e-2, 6.4e-3, 4.8e-3, 5.0e-6])
    eps_s = np.array([1.07e-4, 1.21e-2, 6.3e-3, 3.4e-3, 1.1e-6])
    eps_F = np.array([3.08e-2, 1.83e-2, 8.8e-3, 5.5e-3, 4.5e-5])
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.semilogy(n, eps_F, color=AQUA, marker='s', ms=5, label=r'$\varepsilon_F(n)$: (F) certificada')
    ax.semilogy(n, eps_n, color=AZUL, marker='o', ms=5, label=r'$\varepsilon_n$: Teorema principal')
    ax.semilogy(n, eps_s, color=NARANJA, marker='v', ms=5, ls='--', label=r'$\varepsilon_n^*$: sensibilidad identificada')
    ax.semilogy([6, 9], [3e-3, 9e-2], ls='none', marker='D', ms=6, mfc='white', mec=AMARILLO, mew=1.4,
                label='la ED sugiere')
    ax.semilogy([6, 9], [0.30, 0.29], ls='none', marker='*', ms=9, color=TINTA2,
                label=r'radio de convergencia de $\lambda_0$ (estimado)')
    ax.set_xticks(n)
    ax.set_xlabel(r'$n$')
    ax.set_ylabel(r'$\varepsilon$')
    ax.set_ylim(1e-8, 1)
    ax.set_xlim(5, 19)
    rejilla(ax, 'y')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'fig_radios.pdf'))


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    if len(sys.argv) > 2 and sys.argv[1] == 'deficit':
        deficit_serie(sys.argv[2])
    fig_rangos()
    fig_deficit()
    fig_radios()
    print('escritas', ', '.join(sorted(f for f in os.listdir(FIGS) if f.endswith('.pdf'))))
