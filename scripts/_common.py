"""Utilidades comunes de los guiones de verificación.

Se importa desde los guiones de scripts/ (que se ejecutan como
`python scripts/<nombre>.py` desde la raíz del repositorio, de modo que
scripts/ está en sys.path).
"""
import os

# Hilos de BLAS/OpenMP: se fija un valor por defecto antes de importar numpy
# si el usuario no lo ha fijado (los guiones importan este módulo primero).
os.environ.setdefault("OMP_NUM_THREADS", "2")

import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402

# Los avisos de CVXPY ("Solution may be inaccurate") se registran en la salida
# a través del estado devuelto por el solver; se silencian para no ensuciarla.
warnings.filterwarnings("ignore", message="Solution may be inaccurate")
warnings.filterwarnings("ignore", message="Initializing a Constant with a nested list")

# Tolerancias de Clarabel (las del artículo: 1e-10).
CLARABEL_OPTS = dict(tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)

# Tabla tab:valores del artículo (4 decimales), Theta = 0, pi/4, pi/2, 3pi/4, pi.
ANGLES = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi)
ANGLE_LABELS = ("0", "pi/4", "pi/2", "3pi/4", "pi")
TABLE = {
    3: (0.3333, 0.3333, 0.3333, 0.3333, 0.3333),
    4: (0.6667, 0.5912, 0.5395, 0.5097, 0.5000),
    5: (0.4981, 0.5022, 0.5143, 0.5335, 0.5578),
    6: (0.6228, 0.5964, 0.5770, 0.5655, 0.5617),
    7: (0.5438, 0.5461, 0.5530, 0.5643, 0.5794),
    8: (0.6085, 0.5950, 0.5847, 0.5785, 0.5765),
}
TABLE_TOL = 5e-5 + 1e-7  # media unidad en el cuarto decimal


class Checker:
    """Acumula comprobaciones e imprime una línea "OK"/"FALLO" por cada una."""

    def __init__(self, name):
        self.name = name
        self.failures = 0
        self.count = 0
        self.t0 = time.time()
        self.inaccurate = 0

    def check(self, ok, msg):
        self.count += 1
        if not ok:
            self.failures += 1
        print(f"{'OK   ' if ok else 'FALLO'} {msg}", flush=True)
        return ok

    def info(self, msg):
        print(f"      {msg}", flush=True)

    def status_note(self, status):
        """Texto a añadir a una línea si el solver no terminó en 'optimal'."""
        if status == "optimal":
            return ""
        self.inaccurate += 1
        return f"  [estado: {status}]"

    def finish(self):
        dt = time.time() - self.t0
        print(f"--- {self.name}: {self.count - self.failures}/{self.count} OK, "
              f"{self.failures} FALLO, {self.inaccurate} con estado no 'optimal', "
              f"{dt:.1f} s", flush=True)
        sys.exit(1 if self.failures else 0)


def uniform_gauge(n, Theta):
    """R_e = R_z(Theta/n) en las n aristas i -> i+1 (incluida la de cierre)."""
    from holonomy.paulis import Rz
    return [Rz(Theta / n)] * n


def edge_operators(n, i):
    """A_i = -(XX+YY), B_i = XY-YX, C_i = -ZZ en la arista (i, i+1 mod n)."""
    from holonomy.paulis import pauli_string
    j = (i + 1) % n
    XX = pauli_string(n, ((i, 0), (j, 0)))
    YY = pauli_string(n, ((i, 1), (j, 1)))
    XY = pauli_string(n, ((i, 0), (j, 1)))
    YX = pauli_string(n, ((i, 1), (j, 0)))
    ZZ = pauli_string(n, ((i, 2), (j, 2)))
    return -(XX + YY), XY - YX, -ZZ


def edge_mismatch(rho, n, v, Rs):
    """Error máximo entre las 15 correlaciones de cada arista de rho y las de
    rho_e(v) = (I + v sum T_ab s_a s_b)/4 con T = -R_e (marginales nulas)."""
    from holonomy.paulis import edge_correlations
    err = 0.0
    for i in range(n):
        j = (i + 1) % n
        T, mi, mj = edge_correlations(rho, n, i, j)
        err = max(err, np.max(np.abs(T + v * np.asarray(Rs[i]))),
                  np.max(np.abs(mi)), np.max(np.abs(mj)))
    return err


def u1_eigs(blocks):
    """Autovalores (por bloque y en total) de la rho diagonal por bloques."""
    per = [np.sort(np.linalg.eigvalsh((b + b.conj().T) / 2))[::-1] for b in blocks]
    allw = np.sort(np.concatenate(per))[::-1]
    return per, allw
