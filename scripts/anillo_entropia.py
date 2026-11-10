"""Entropía del estado pegado óptimo del anillo de 6 frente a la holonomía: marca la ventana de desanclaje.

Artículo II (~/fisica/holonomia_fisher.tex), sección del signo de la holonomía (anillos), párrafo «La
entropía residual marca la ventana». Para cada Theta:
  1. Testigo óptimo por el dual del anillo (holonomy.dual: flujo libre, y la cúspide Phi = pi si el flujo óptimo está
     en ella), construido como matriz de 2^6 en la base de qubits, W = sum_e (cos chi A_e + sin chi B_e + Delta C_e),
     chi = Phi/n; se comprueba lambda_max(W)/N(W) = vmax.
  2. Primal por mínimos cuadrados dentro del autoespacio máximo de W, con las 15 correlaciones de cada arista y la
     traza (holgura complementaria: todo óptimo vive ahí). Rango completo -> rho* único.
  3. S(rho*) y, fuera de la ventana, el peso p_- de la rama de corriente opuesta frente a la fórmula (eq:pesos) del
     Teorema de la cúspide del primer artículo, p_-+ = 1/2 -+ d_Delta eps sin d / (n s_n).
  4. Cerca del borde, S ~ p_-(1 - ln p_-) y p_- lineal en la distancia al borde.
  5. S(rho*) = lim S_G(v) cuando v -> vmax: el estado de máxima entropía (fisher_kubo_mori.MaxEnt) en 1 - v/vmax = 1e-4.

Escribe data/anillo_entropia.csv y data/anillo_entropia_limite.csv.

Uso: (ulimit -v 8000000; OMP_NUM_THREADS=4 .venv/bin/python scripts/anillo_entropia.py)
"""
import _common  # noqa: F401
from _common import Checker

import numpy as np

from holonomy import bethe, dual, xxz
from holonomy.paulis import pauli_string, Rz
from holonomy.report import write_csv
from fisher_kubo_mori import MaxEnt

N = 6
PSI_PUBLICADO = 0.018587  # tabla de la anchura de la ventana del primer artículo, n = 6 (psi/pi)
DENTRO = (0.005, 0.01, 0.015)
LEJOS = (0.0195, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0)
CERCA = (1e-4, 3e-4, 1e-3, 3e-3)  # distancia al borde, en unidades de pi
PESOS = (0.1, 0.25, 0.5, 0.75, 1.0)  # Theta/pi donde se contrasta (eq:pesos) con el primal


def mat(o):
    return np.asarray(o.todense() if hasattr(o, "todense") else o, dtype=complex)


A = [mat(-(pauli_string(N, ((i, 0), ((i + 1) % N, 0))) + pauli_string(N, ((i, 1), ((i + 1) % N, 1))))) for i in range(N)]
B = [mat(pauli_string(N, ((i, 0), ((i + 1) % N, 1))) - pauli_string(N, ((i, 1), ((i + 1) % N, 0)))) for i in range(N)]
C = [mat(-pauli_string(N, ((i, 2), ((i + 1) % N, 2)))) for i in range(N)]


def testigo(Theta, chi, Delta):
    W = sum(np.cos(chi) * a + np.sin(chi) * b + Delta * c for a, b, c in zip(A, B, C))
    return W, N * (Delta + 2 * np.cos(chi - Theta / N))


def optimo(Theta):
    """(v, Phi, Delta, en_cuspide) del dual."""
    v, P, D = dual.v_free(xxz.eps, N, Theta)
    vc, Dc = dual.v_cusp(xxz.eps, N, Theta)
    if vc <= v + 1e-12:
        return vc, dual.cusp_flux(N), Dc, True
    return v, P, D, False


def primal(Theta, v, W):
    """rho* por mínimos cuadrados en el autoespacio máximo de W: (rho, grado, rango, residuo)."""
    E, V = np.linalg.eigh(W)
    d = int((E > E[-1] - 1e-8).sum())
    Vt = V[:, -d:]
    # base real de las hermíticas d x d
    base = []
    for a in range(d):
        for b in range(a, d):
            M = np.zeros((d, d), complex)
            M[a, b] = M[b, a] = 1
            base.append(M)
            if a != b:
                M = np.zeros((d, d), complex)
                M[a, b], M[b, a] = 1j, -1j
                base.append(M)
    ops, objetivo = [np.eye(2**N)], [1.0]
    for i in range(N):
        j = (i + 1) % N
        T = -v * Rz(Theta / N)
        for a in range(3):
            ops.append(mat(pauli_string(N, ((i, a),))))
            objetivo.append(0.0)
            for b in range(3):
                ops.append(mat(pauli_string(N, ((i, a), (j, b)))))
                objetivo.append(T[a, b])
    red = [Vt.conj().T @ O @ Vt for O in ops]
    X = np.array([[np.trace(R @ M).real for M in base] for R in red])
    coef, *_ = np.linalg.lstsq(X, np.array(objetivo), rcond=None)
    rango = np.linalg.matrix_rank(X, tol=1e-10)
    M = sum(c * m for c, m in zip(coef, base))
    return Vt @ M @ Vt.conj().T, d, rango, float(np.abs(X @ coef - objetivo).max()), len(base)


def entropia(p):
    p = p[p > 1e-15]
    return float(-(p * np.log(p)).sum())


def p_pesos(Theta, Delta):
    """p_- de (eq:pesos): 1/2 - d_Delta eps sin d / (n s_n), en la cúspide."""
    Pc, h = dual.cusp_flux(N), 1e-5
    dDe = (xxz.eps(N, Delta + h, Pc) - xxz.eps(N, Delta - h, Pc)) / (2 * h)
    d = abs(Pc - Theta) / N
    return 0.5 - dDe * np.sin(d) / (N * bethe.cusp_slope(N, Delta))


def main():
    chk = Checker("anillo_entropia")
    psi, Dborde = dual.window_edge(xxz.eps, N)
    chk.check(abs(psi / np.pi - PSI_PUBLICADO) < 5e-7, f"n = 6: borde de la ventana en Theta = {psi / np.pi:.7f} pi "
                                                       f"(tabla: {PSI_PUBLICADO})")
    filas = []
    thetas = sorted(set(DENTRO) | set(LEJOS) | {psi / np.pi + c for c in CERCA})
    datos = {}
    for tp in thetas:
        Theta = tp * np.pi
        v, P, D, cusp = optimo(Theta)
        W, Nw = testigo(Theta, P / N, D)
        lam = np.linalg.eigvalsh(W)[-1]
        rho, d, rango, res, npar = primal(Theta, v, W)
        p = np.clip(np.linalg.eigvalsh(rho), 0, None)
        p = p / p.sum()  # la traza se ajusta por mínimos cuadrados, a 1e-8; S es la de un estado
        S = entropia(p)
        pm = float(p[-2]) if d == 2 else 0.0
        pp = p_pesos(Theta, D) if cusp else np.nan
        chk.check(abs(lam / Nw - v) < 1e-9 and res < 1e-7 and rango == npar and np.linalg.eigvalsh(rho)[0] > -1e-9,
                  f"Theta = {tp:.6f} pi: vmax = {v:.10f} = lambda_max/N ({lam / Nw - v:+.1e}); {'cúspide' if cusp else 'flujo libre'}, "
                  f"autoespacio {d}, rango {rango}/{npar}, residuo {res:.1e}; S = {S:.6f}, p_- = {pm:.3e}"
                  + (f", (eq:pesos) {pp:.3e}" if cusp else ""))
        datos[tp] = (v, P, D, cusp, d, rango, res, pm, pp, S)
        filas.append([tp, v, P, D, int(cusp), d, rango, res, pm, pp, S, pm * (1 - np.log(pm)) if pm > 0 else 0.0])
    write_csv("anillo_entropia.csv", ("Theta_pi", "vmax", "Phi", "Delta", "en_cuspide", "dim_autoespacio", "rango",
                                      "residuo", "p_menos", "p_menos_pesos", "S", "S_aprox"), filas)

    dentro = [datos[t] for t in DENTRO]
    chk.check(all(not x[3] and x[4] == 1 and x[9] < 1e-12 for x in dentro),
              "dentro de la ventana: flujo fuera de la cúspide, autovalor máximo simple, rho* puro, S = 0")
    fuera = [datos[t] for t in thetas if t > psi / np.pi]
    chk.check(all(x[3] and x[4] == 2 for x in fuera), "fuera de la ventana: cúspide y doblete en todos los puntos")
    err = max(abs(datos[t][7] - datos[t][8]) for t in PESOS)
    chk.check(err < 1e-7, f"p_- del primal frente a (eq:pesos) en Theta/pi = {PESOS}: error máximo {err:.1e}")
    chk.check(abs(datos[1.0][9] - np.log(2)) < 1e-9, f"Theta = pi: S = {datos[1.0][9]:.10f} = ln 2, pesos 1/2")
    cerca = [datos[psi / np.pi + c] for c in CERCA]
    pend = [x[7] / c for x, c in zip(cerca, CERCA)]
    chk.check(max(pend) / min(pend) < 1.05, f"cerca del borde p_- es casi lineal: p_-/(Theta - borde) en pi = "
                                            f"{', '.join(f'{z:.3f}' for z in pend)}")
    rat = [x[9] / (x[7] * (1 - np.log(x[7]))) for x in cerca]
    chk.check(abs(rat[0] - 1) < 1e-3 and abs(rat[0] - 1) < abs(rat[-1] - 1),
              f"S / [p_-(1 - ln p_-)] = {', '.join(f'{z:.5f}' for z in rat)} -> 1 en el borde: S ~ x ln(1/x)")

    # S(rho*) como entropía residual: límite del estado de máxima entropía
    me = MaxEnt(N, [(i, (i + 1) % N) for i in range(N)])
    lim = []
    for tp in (0.01, 0.25, 1.0):
        v = datos[tp][0]
        x = np.zeros(len(me.O))
        for eps in (0.1, 1e-2, 1e-3, 1e-4, 1e-5):
            x, g, SG, r = me.resuelve(me.objetivo(v * (1 - eps), np.full(N, tp * np.pi / N)), x)
        chk.check(r < 1e-10 and abs(SG - datos[tp][9]) < 5e-3,
                  f"Theta = {tp} pi: S_G(v) en 1 - v/vmax = 1e-5 es {SG:.5f}, S(rho*) = {datos[tp][9]:.5f} "
                  f"(residuo {r:.1e})")
        lim.append([tp, 1e-5, SG, datos[tp][9], r])
    write_csv("anillo_entropia_limite.csv", ("Theta_pi", "eps", "S_G", "S_optimo", "residuo"), lim)
    chk.finish()


if __name__ == "__main__":
    main()
