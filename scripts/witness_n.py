"""Testigo de n sitios por sectores de magnetización (secciones sec:nsitios y sec:reduccion).

Para n = 3..8 y Theta = 0, pi/4, pi/2, 3pi/4, pi:
  1. valor dual (Proposición prop:xxz) con holonomy.dual.v_cusp, v_smooth y
     v_free sobre holonomy.xxz.eps (diagonalización por sectores de
     magnetización); se comprueba que v_free = min(v_cusp, v_smooth) en estos
     ángulos y que eps (sectores de magnetización mínima, más el ferromagnético)
     coincide con eps_full (todos los sectores) en el óptimo;
  2. comparación con la tabla tab:valores (4 decimales);
  3. comparación con el primal (artículo: <= 1e-6):
       n <= 5: SDP primal completo (holonomy.sdp.primal);
       n = 6, 7: primal reducido por U(1) (holonomy.sdp.primal_u1);
       todos los n: primal reducido por U(1) y traslación, implementado aquí
       (bloques de número de giros k y momento q). Es el único primal para
       n = 8: primal_u1 con n = 8 tarda ~7 min por ángulo con Clarabel en esta
       máquina. En n <= 7 se valida contra los otros primales.

Uso: python scripts/witness_n.py
"""
import _common  # noqa: F401  (fija OMP_NUM_THREADS antes de numpy)
from _common import (ANGLE_LABELS, ANGLES, CLARABEL_OPTS, TABLE, TABLE_TOL, Checker,
                     edge_operators, uniform_gauge)

import cvxpy as cp
import numpy as np

from holonomy import dual, xxz
from holonomy.paulis import pauli_string
from holonomy.sdp import primal, primal_u1

TOL = 1e-6


def _translation(n):
    """Permutación de la base computacional: el qubit i pasa al sitio i+1."""
    perm = np.empty(2 ** n, dtype=int)
    for s in range(2 ** n):
        bits = [(s >> (n - 1 - i)) & 1 for i in range(n)]
        new = [bits[(i - 1) % n] for i in range(n)]
        perm[s] = int("".join(map(str, new)), 2)
    return perm


def symmetric_blocks(n):
    """Isometrías V_{k,q} (sector de k giros, momento 2 pi q/n) y operadores
    totales A, B, C, Z reducidos a cada bloque."""
    A = sum(edge_operators(n, i)[0] for i in range(n))
    B = sum(edge_operators(n, i)[1] for i in range(n))
    C = sum(edge_operators(n, i)[2] for i in range(n))
    Zt = sum(pauli_string(n, ((i, 2),)) for i in range(n))
    perm = _translation(n)
    out = []
    for k in range(n + 1):
        idx = [s for s in range(2 ** n) if bin(s).count("1") == k]
        pos = {s: m for m, s in enumerate(idx)}
        dim = len(idx)
        P = np.zeros((dim, dim))
        for m, s in enumerate(idx):
            P[pos[perm[s]], m] = 1.0
        Pm = [np.linalg.matrix_power(P, m) for m in range(n)]
        for q in range(n):
            w = np.exp(-2j * np.pi * q / n)
            proj = sum(w ** m * Pm[m] for m in range(n)) / n
            ev, vec = np.linalg.eigh((proj + proj.conj().T) / 2)
            V = vec[:, ev > 0.5]
            if V.shape[1] == 0:
                continue
            red = {name: V.conj().T @ M[np.ix_(idx, idx)] @ V
                   for name, M in (("A", A), ("B", B), ("C", C), ("Z", Zt))}
            out.append((k, q, V, red))
    return out


def primal_symmetric(n, Theta, blocks):
    """max v con rho = suma directa de bloques X_{k,q} >= 0 y las restricciones
    sumadas sobre aristas (equivalentes por invariancia traslacional)."""
    phi = Theta / n
    v = cp.Variable()
    cons, lin = [], {"A": 0, "B": 0, "C": 0, "Z": 0, "tr": 0}
    for _, _, V, red in blocks:
        d = V.shape[1]
        Xb = cp.Variable((d, d), hermitian=True)
        cons.append(Xb >> 0)
        lin["tr"] += cp.real(cp.trace(Xb))
        for name in ("A", "B", "C", "Z"):
            lin[name] += cp.real(cp.sum(cp.multiply(red[name].T, Xb)))
    cons += [lin["tr"] == 1,
             lin["A"] == 2 * n * v * np.cos(phi),
             lin["B"] == 2 * n * v * np.sin(phi),
             lin["C"] == n * v,
             lin["Z"] == 0]
    prob = cp.Problem(cp.Maximize(v), cons)
    prob.solve(solver="CLARABEL", **CLARABEL_OPTS)
    return float(v.value), prob.status


def main():
    chk = Checker("witness_n")
    for n in range(3, 9):
        print(f"== n = {n}")
        blocks = symmetric_blocks(n)
        for j, (Theta, lab) in enumerate(zip(ANGLES, ANGLE_LABELS)):
            vc, Dc = dual.v_cusp(xxz.eps, n, Theta)
            vs, Ds = dual.v_smooth(xxz.eps, n, Theta)
            vf, Pf, Df = dual.v_free(xxz.eps, n, Theta)
            tag = f"n={n} Theta={lab:5s}"
            chk.check(abs(vf - min(vc, vs)) < 1e-9 and vf <= min(vc, vs) + 1e-12,
                      f"{tag}: dual v_free={vf:.10f} (Phi*/pi={Pf / np.pi:.4f}, Delta*={Df:.4f}); "
                      f"v_cusp={vc:.10f} (Delta={Dc:.4f}); v_smooth={vs:.10f} (Delta={Ds:.4f}); "
                      f"|v_free-min|={abs(vf - min(vc, vs)):.1e}")
            if np.isfinite(Df):
                e, ef = xxz.eps(n, Df, Pf), xxz.eps_full(n, Df, Pf)
                chk.check(abs(e - ef) < 1e-12, f"{tag}: eps por sectores = eps_full en el óptimo "
                          f"(|dif|={abs(e - ef):.1e})")
            else:
                chk.info(f"{tag}: Delta* = inf (límite de Ising, dual.ising_limit = {dual.ising_limit(n):.10f})")
            ref = TABLE[n][j]
            chk.check(abs(vf - ref) < TABLE_TOL, f"{tag}: dual vs tabla {ref:.4f}: |dif|={abs(vf - ref):.1e}")

            primals = []
            if n <= 5:
                vp, _, st = primal(n, uniform_gauge(n, Theta), return_state=True, **CLARABEL_OPTS)
                primals.append(("primal completo", vp, st))
            if n in (6, 7):
                vp, _, st = primal_u1(n, Theta, return_blocks=True, **CLARABEL_OPTS)
                primals.append(("primal_u1", vp, st))
            vp, st = primal_symmetric(n, Theta, blocks)
            primals.append(("primal U(1)+traslación", vp, st))
            for name, vp, st in primals:
                chk.check(abs(vp - vf) < TOL, f"{tag}: {name}={vp:.10f} vs dual |dif|={abs(vp - vf):.1e}"
                          f"{chk.status_note(st)}")
    chk.finish()


if __name__ == "__main__":
    main()
