"""Criterio a priori del defecto ganador: energía del defecto a segundo orden (exploración).

Para el término aleatorio de celda_k_aleatorio.py (h = -P_asc + eps K + delta D), y en cada sector de carga S^z con
estados clásicos de un defecto, se calcula el hamiltoniano efectivo de Schrieffer-Wolff a segundo orden sobre esos
estados,
    H_eff = P V P + P V Q (E_cl - H_cl)^{-1} Q V P,     V = H - H_cl,
y su mínimo, E_def(S^z). El criterio predice que gana el sector de menor E_def. Se compara con la diagonalización
exacta, en el mismo n y en n + 3, en Theta = 0 o (TODO_THETA=1) con el mínimo sobre Theta, para ver si el criterio, calculado en un anillo pequeño, sirve en
uno mayor.

Uso: python scripts/criterio_defecto.py [muestras] [eps] [delta] [semilla]   (NS en el entorno, p. ej. "7,8")
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

from celda_k_aleatorio import Anillo, termino  # noqa: E402


def clasico():
    h = np.zeros((9, 9), complex)
    for a in range(3):
        h[3 * a + (a + 1) % 3, 3 * a + (a + 1) % 3] = -1
    return h


def energia_defecto(A, h, Sz, theta=0.0):
    n = A.n
    H = A.hamiltoniano(h, Sz, theta)
    H0 = np.real(A.hamiltoniano(clasico(), Sz, 0.0).diagonal())
    Ecl = -(n - 1)
    P = np.nonzero(np.isclose(H0, Ecl))[0]
    if P.size == 0:
        return None, 0
    Qm = ~np.isclose(H0, Ecl)
    V = (H - sp.diags(H0)).tocsr()
    VPP = V[P][:, P].toarray()
    VQP = V[Qm][:, P]
    den = 1.0 / (Ecl - H0[Qm])
    Heff = VPP + (VQP.getH() @ sp.diags(den) @ VQP).toarray()
    Heff = 0.5 * (Heff + Heff.conj().T)
    return Ecl + np.linalg.eigvalsh(Heff)[0], P.size


# Con TODO_THETA=1 se compara el mínimo sobre Theta de cada sector (el mínimo de la banda del defecto), que es lo que
# decide el ganador a tamaño grande; por defecto, solo Theta = 0.
THETAS = np.linspace(0, 2 * np.pi, 9)[:-1] if os.environ.get("TODO_THETA") == "1" else [0.0]


def main():
    muestras = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    eps = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
    delta = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
    semilla = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    ns = [int(x) for x in os.environ.get("NS", "7,8").split(",")]
    anillos = {}
    for n in ns:
        anillos[n] = Anillo(n)
        anillos[n + 3] = Anillo(n + 3)
    rng = np.random.default_rng(semilla)
    print(f"eps={eps} delta={delta} semilla={semilla}")
    aciertos = {n: [0, 0] for n in ns}
    for s in range(muestras):
        h = termino(rng, eps, delta)
        for n in ns:
            A, B = anillos[n], anillos[n + 3]
            crit = {}
            for Sz in (-1, 0, 1):
                es = [energia_defecto(A, h, Sz, t)[0] for t in THETAS]
                if es[0] is not None:
                    crit[Sz] = min(es)
            ed_n = {Sz: min(A.e0(h, Sz, t) for t in THETAS) for Sz in crit}
            ed_N = {Sz: min(B.e0(h, Sz, t) for t in THETAS) for Sz in crit}
            g_c, g_n, g_N = (min(d, key=d.get) for d in (crit, ed_n, ed_N))
            orden = sorted(crit, key=crit.get)
            margen = crit[orden[1]] - crit[orden[0]]
            orden_ed = sorted(ed_N, key=ed_N.get)
            margen_ed = ed_N[orden_ed[1]] - ed_N[orden_ed[0]]
            ok = g_c == g_N
            aciertos[n][0] += ok
            aciertos[n][1] += 1
            print(f"[{s:2d}] n={n}: criterio Sz={g_c:+d} (margen {margen:.2e}) | ED n={n}: {g_n:+d} | "
                  f"ED n={n + 3}: {g_N:+d} (margen {margen_ed:.2e}) {'ok' if ok else 'FALLO'}", flush=True)
    for n in ns:
        print(f"n={n} -> n={n + 3}: aciertos {aciertos[n][0]}/{aciertos[n][1]}")


if __name__ == "__main__":
    main()
