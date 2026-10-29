"""Escaleras de dos patas (exploración del problema abierto de grafos con varios ciclos).

  1. Escalera abierta 2x3 (6 qubits, 7 aristas, 2 plaquetas) con el SDP primal
     completo (holonomy.sdp.primal): holonomías abelianas e iguales, de signo
     opuesto o en una sola plaqueta, y no abelianas (R_z y R_x). Gauge de
     árbol: toda la holonomía en las aristas de la pata superior.
  2. Cotas que no dependen de la escalera: la estrella K_{1,3} de un sitio de
     grado 3 (un árbol no tiene holonomía) da v <= 5/9, y cada plaqueta es un
     anillo de 4, v <= 2/(3 + sin(f_p/2)) (Teorema 1).
  3. El primal reducido por U(1) (holonomy.sdp.primal_u1_graph) reproduce el
     completo en los casos abelianos de 1 (<= 1e-6).
  4. Escalera abierta 2x4 con flujo uniforme (primal U(1)).
  5. Escalera periódica con flujo uniforme: el primal reducido por U(1) y
     traslación (holonomy.ladder.primal_symmetric) coincide con el reducido
     solo por U(1) en 2x4, y el dual reducido a ocho coeficientes
     (holonomy.ladder.LadderDual) coincide con el primal en 2x4 y 2x5
     (<= 1e-6): basta el testigo simétrico.
  6. Barrido del dual en el flujo para 2x4, 2x5 y 2x6.

Escribe data/ladder.csv.

Uso: python scripts/ladder.py
"""
import _common  # noqa: F401  (fija OMP_NUM_THREADS antes de numpy)
from _common import CLARABEL_OPTS, Checker

import numpy as np

from holonomy import ladder
from holonomy.paulis import Rx, Rz
from holonomy.report import write_csv
from holonomy.sdp import primal, primal_u1_graph

TOL = 1e-6
STAR = 5 / 9


def f4(theta):
    return 2 / (3 + np.sin(theta / 2))


def tree_gauge_open(L, tops):
    """Matrices R_e en el orden de ladder_edges(L): identidad salvo en la pata
    superior, donde va la holonomía de cada plaqueta (recorrida p -> p+1 ->
    L+p+1 -> L+p, la arista superior se recorre al revés: holonomía R^T)."""
    m = L - 1
    return [np.eye(3)] * m + list(tops) + [np.eye(3)] * L


def main():
    chk = Checker("ladder")
    rows = []
    L, n = 3, 6
    edges = ladder.ladder_edges(L)
    pi = np.pi

    print("== Escalera abierta 2x3, primal completo")
    # (nombre, R_A, R_B, ángulos de plaqueta, abeliano)
    cases = [(f"({t / pi:.2f}pi, {t / pi:.2f}pi)", Rz(t), Rz(t), (t, t), True) for t in
             (0, pi / 4, pi / 2, 3 * pi / 4, pi)]
    for t in (pi / 2, pi):
        cases += [(f"({t / pi:.2f}pi, 0)", Rz(t), np.eye(3), (t, 0.0), True),
                  (f"({t / pi:.2f}pi, -{t / pi:.2f}pi)", Rz(t), Rz(-t), (t, -t), True),
                  (f"Rz({t / pi:.2f}pi), Rx({t / pi:.2f}pi)", Rz(t), Rx(t), (t, t), False)]
    for name, RA, RB, flux, abelian in cases:
        v, _, st = primal(n, tree_gauge_open(L, (RA, RB)), edges=edges, return_state=True,
                          **CLARABEL_OPTS)
        tag = f"2x3 abierta {name}"
        chk.info(f"{tag}: v_max = {v:.8f}{chk.status_note(st)}")
        bound = min(STAR, *(f4(abs(f)) for f in flux))
        chk.check(v <= bound + TOL, f"{tag}: v <= min(5/9, cota de plaqueta) = {bound:.6f}")
        if abelian:
            # La holonomía de la plaqueta es R^T de la arista superior: flujo -t;
            # en el gauge simétrico se escribe directamente con el flujo -t.
            ph = ladder.abelian_phases(L, [-f for f in flux])
            vu, _, stu = primal_u1_graph(n, edges, ph, return_blocks=True, **CLARABEL_OPTS)
            chk.check(abs(vu - v) < TOL, f"{tag}: primal U(1) = {vu:.8f}, |dif| = {abs(vu - v):.1e}"
                      f"{chk.status_note(stu)}")
        rows.append(("abierta", L, name, v))
    v0 = rows[0][3]
    chk.check(abs(v0 - STAR) < TOL, f"2x3 abierta sin holonomía alcanza 5/9: |dif| = {abs(v0 - STAR):.1e}")

    print("== Escalera abierta 2x4, primal U(1)")
    L = 4
    for t, lab in ((pi / 2, "pi/2"), (pi, "pi")):
        v, _, st = primal_u1_graph(2 * L, ladder.ladder_edges(L), ladder.abelian_phases(L, t),
                                   return_blocks=True, **CLARABEL_OPTS)
        chk.info(f"2x4 abierta flujo uniforme {lab}: v_max = {v:.8f}{chk.status_note(st)}")
        chk.check(v <= min(STAR, f4(t)) + TOL, f"2x4 abierta {lab}: v <= min(5/9, cota de plaqueta)")
        rows.append(("abierta", L, f"uniforme {lab}", v))

    print("== Escalera periódica, flujo uniforme: dual reducido frente a primal")
    # 2x4: el primal U(1)+traslación frente al primal U(1) sin traslación.
    ph = ladder.abelian_phases(4, pi / 2, periodic=True)
    vs, st = ladder.primal_symmetric(4, ph, **CLARABEL_OPTS)
    vu, _, stu = primal_u1_graph(8, ladder.ladder_edges(4, True), ph, return_blocks=True, **CLARABEL_OPTS)
    chk.check(abs(vs - vu) < TOL, f"2x4 periódica pi/2: primal U(1)+traslación = {vs:.8f}, "
              f"primal U(1) = {vu:.8f}, |dif| = {abs(vs - vu):.1e}{chk.status_note(stu)}")
    for L, angles in ((4, ((0.0, "0"), (pi / 2, "pi/2"), (pi, "pi"))), (5, ((pi / 2, "pi/2"),))):
        D = ladder.LadderDual(L)
        for t, lab in angles:
            ph = ladder.abelian_phases(L, t, periodic=True)
            f, legs = ladder.plaquette_fluxes(L, ph, periodic=True)
            chk.check(np.allclose(f, t), f"2x{L} periódica f={lab}: flujos de plaqueta {np.round(f, 12)}, "
                      f"patas {np.round(legs, 12)}")
            vd, y = D.v_dual(ph)
            vp, st = ladder.primal_symmetric(L, ph, **CLARABEL_OPTS)
            chk.check(abs(vd - vp) < TOL, f"2x{L} periódica f={lab}: dual = {vd:.8f}, primal = {vp:.8f}, "
                      f"|dif| = {abs(vd - vp):.1e}{chk.status_note(st)}")
            chk.info(f"   testigo (alpha_b, beta_b, c_b, alpha_t, beta_t, c_t, alpha_r, c_r) = "
                     f"{np.round(y / np.max(np.abs(y)), 4)}")
            chk.check(vp <= min(STAR, f4(t)) + TOL, f"2x{L} periódica f={lab}: v <= min(5/9, cota de plaqueta)")

    print("== Barrido del dual, escalera periódica con flujo uniforme (cota superior; igual al primal donde se contrastó)")
    for L in (4, 5, 6):
        D = ladder.LadderDual(L)
        for t in np.linspace(0, pi, 9):
            vd, _ = D.v_dual(ladder.abelian_phases(L, t, periodic=True))
            chk.info(f"2x{L} periódica f/pi={t / pi:.3f}: dual = {vd:.8f}")
            rows.append(("periodica (dual)", L, f"uniforme {t / pi:.3f}pi", vd))

    write_csv("ladder.csv", ["contorno", "L", "holonomia", "v_max"], rows)
    chk.finish()


if __name__ == "__main__":
    main()
