"""DMRG del anillo de celda 3 con flujo, en un sector de S^z fijo (exploración; requiere TeNPy).

Mismo hamiltoniano que celda_k_aleatorio.py: espín 1, s in {0,1,2} con S^z = s - 1, término de enlace h (9x9) y giro
en el segundo sitio, <a'b'|h^Theta|ab> = e^{-i phi (q_b' - q_b)} <a'b'|h|ab>, phi = Theta/n, en el gauge uniforme. El
anillo es periódico (MPO con el enlace largo) y la carga S^z se conserva en las MPS.

Uso: python scripts/dmrg_flujo.py semilla eps delta muestra Sz n [n ...]   (CHI y NTHETA en el entorno)
"""
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
import logging  # noqa: E402

import numpy as np  # noqa: E402
from tenpy.algorithms import dmrg  # noqa: E402
from tenpy.linalg import np_conserved as npc  # noqa: E402
from tenpy.models.model import CouplingMPOModel  # noqa: E402
from tenpy.networks.mps import MPS  # noqa: E402
from tenpy.networks.site import Site  # noqa: E402

from celda_k_aleatorio import Q, termino  # noqa: E402

logging.getLogger("tenpy").setLevel(logging.WARNING)


def sitio():
    leg = npc.LegCharge.from_qflat(npc.ChargeInfo([1], ["Sz"]), [[-1], [0], [1]])
    s = Site(leg, ["m", "0", "p"])
    for x in range(3):
        for y in range(3):
            E = np.zeros((3, 3))
            E[x, y] = 1
            s.add_op(f"E{x}{y}", E)
    return s


class Anillo(CouplingMPOModel):
    def init_sites(self, model_params):
        return sitio()

    def init_terms(self, model_params):
        h = model_params.get("h", None)
        n = model_params.get("L", None)
        theta = model_params.get("theta", 0.0)
        phi = theta / n
        for x in range(9):  # entrada (a, b)
            a, b = divmod(x, 3)
            for y in range(9):  # salida (a', b')
                v = h[y, x]
                if abs(v) < 1e-15:
                    continue
                a2, b2 = divmod(y, 3)
                v = v * np.exp(-1j * phi * (Q[b2] - Q[b]))
                self.add_coupling(v, 0, f"E{a2}{a}", 0, f"E{b2}{b}", 1)


def energia(h, n, theta, Sz, chi, psi0=None):
    M = Anillo(dict(L=n, h=h, theta=theta, lattice="Chain", bc_x="periodic", bc_MPS="finite"))
    if psi0 is None:
        conf = [i % 3 for i in range(n)]
        while sum(conf) - n != Sz:  # ajusta la carga cambiando sitios
            k = conf.index(0) if sum(conf) - n > Sz else conf.index(2)
            conf[k] += -1 if sum(conf) - n > Sz else 1
        psi0 = MPS.from_product_state(M.lat.mps_sites(), [["m", "0", "p"][c] for c in conf], bc="finite")
    psi = psi0.copy()
    info = dmrg.run(psi, M, {"mixer": True, "trunc_params": {"chi_max": chi, "svd_min": 1e-12},
                             "max_E_err": 1e-12, "min_sweeps": 6, "max_sweeps": 60, "combine": True})
    return info["E"], psi


def main():
    semilla, eps, delta, muestra, Sz = (int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]),
                                         int(sys.argv[4]), int(sys.argv[5]))
    ns = [int(x) for x in sys.argv[6:]]
    chi = int(os.environ.get("CHI", 120))
    nth = int(os.environ.get("NTHETA", 8))
    rng = np.random.default_rng(semilla)
    for _ in range(muestra + 1):
        h = termino(rng, eps, delta)
    thetas = np.linspace(0, 2 * np.pi, nth + 1)[:-1]
    for n in ns:
        t0 = time.time()
        E, psi = [], None
        for t in thetas:
            e, psi = energia(h, n, t, Sz, chi, psi)
            E.append(e)
        E = np.array(E)
        R = E.max() - E.min()
        print(f"muestra {muestra} n={n} Sz={Sz:+d} chi={chi}: R={R:.5e} n2R={n * n * R:.4f} "
              f"argmin/pi={thetas[E.argmin()] / np.pi:.2f} E={' '.join(f'{e:.9f}' for e in E)} "
              f"chi_max_usado={max(psi.chi)} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
