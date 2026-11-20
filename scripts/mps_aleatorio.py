"""¿Es el déficit O(lambda^{2n}) propio de AKLT? MPS aleatorios con U(1) y núcleo.

Con AKLT (scripts/aklt_lanczos.py) el déficit 1 - p(n, Theta) y la sensibilidad van
como lambda^{2n}, el cuadrado de la cota del Teorema teo:mps. Aquí se repite con MPS
aleatorios que cumplen (M): espín 1 (d = 3, cargas +1, 0, -1), D = 2 con cargas
virtuales +1/2 y -1/2. La regla de carga de (M2) deja
    A^+ = a sigma^+,   A^0 = diag(b1, b2),   A^- = c sigma^-,
con a, b1, b2, c complejos; AKLT es a = sqrt(2/3), b1 = -b2 = -sqrt(1/3), c = -sqrt(2/3).
Con D = 2 < d el rango de sigma es <= 4 < 9: hay núcleo y techo p <= 1. Si b1 != -b2
los datos no son invariantes por SU(2) y el canal no es normal en general.

Para cada MPS: canal normalizado (E(I) = I, punto fijo dual Lambda), lambda (segundo
módulo), estructura de Jordan, sigma y su núcleo. El testigo de techo es
h_c = I - (9 / dim ker) (1⊗V) P_ker (1⊗V)^†, con f = 1 (el MPS retorcido lo anula). El
dual se resuelve como en aklt_lanczos.py (Lanczos, Newton con hessiano exacto,
continuación en Theta) y p está certificado por las dos cotas.

Aquí solo se mide el déficit en Theta = 0 y pi. La sensibilidad s = max - min en Theta no se
puede leer de una malla en [0, pi]: la fase de b2/b1 es una espiral que desplaza Theta en
n alpha, así que p no es par en Theta. La sensibilidad se mide en forma canónica real en
scripts/mps_su2.py.

Salida: data/mps_aleatorio.csv. Uso: python scripts/mps_aleatorio.py [nmax].
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from aklt_lanczos import AKLTDual, sector_ops, check  # noqa: E402
import aklt_lanczos  # noqa: E402
from aklt_sdp_ti import M_OF, herm_basis  # noqa: E402

D = 3


def tensors(a, b1, b2, c):
    sp_, sm = np.array([[0, 1], [0, 0]], complex), np.array([[0, 0], [1, 0]], complex)
    return [a * sp_, np.diag([b1, b2]).astype(complex), c * sm]   # m = +1, 0, -1


def normalize(A):
    """Gauge con E(I) = I y punto fijo dual Lambda (traza 1). Devuelve (A, Lambda, espectro)."""
    E = sum(np.kron(As, As.conj()) for As in A)          # vec(X) por filas: A X A^†
    w, V = np.linalg.eig(E)
    k = np.argmax(np.abs(w))
    A = [As / np.sqrt(w[k].real) for As in A]
    E = sum(np.kron(As, As.conj()) for As in A)
    w, V = np.linalg.eig(E)
    k = np.argmax(np.abs(w))
    R = V[:, k].reshape(2, 2)
    R = R / np.trace(R)
    R = (R + R.conj().T) / 2
    Rh = np.linalg.cholesky(R) if np.all(np.linalg.eigvalsh(R) > 0) else None
    ev, U = np.linalg.eigh(R)
    Rs = U @ np.diag(np.sqrt(ev)) @ U.conj().T
    Ri = U @ np.diag(1 / np.sqrt(ev)) @ U.conj().T
    A = [Ri @ As @ Rs for As in A]
    E = sum(np.kron(As, As.conj()) for As in A)
    wl, VL = np.linalg.eig(E.conj().T)
    k = np.argmax(np.abs(wl))
    L = VL[:, k].reshape(2, 2).conj()
    L = L / np.trace(L)
    L = (L + L.conj().T) / 2
    return A, L, np.sort(np.abs(np.linalg.eigvals(E)))[::-1], E


def marginal(A, L):
    """sigma_{ab,cd} = tr(L A^a A^b A^{d†} A^{c†})."""
    s = np.zeros((9, 9), complex)
    for a in range(3):
        for b in range(3):
            for c in range(3):
                for d in range(3):
                    s[3 * a + b, 3 * c + d] = np.trace(L @ A[a] @ A[b] @ A[d].conj().T @ A[c].conj().T)
    return (s + s.conj().T) / 2


def jordan_size(E, lam, tol=1e-8):
    """Tamaño del mayor bloque de Jordan de los autovalores de módulo lam."""
    w = np.linalg.eigvals(E)
    J = 1
    for mu in w[np.abs(np.abs(w) - lam) < 1e-9]:
        N = E - mu * np.eye(len(E))
        r = [np.linalg.matrix_rank(np.linalg.matrix_power(N, k), tol) for k in range(1, 5)]
        J = max(J, 1 + sum(1 for k in range(len(r) - 1) if r[k] > r[k + 1]))
    return J


class MPSDual(AKLTDual):
    def __init__(self, n, Theta, ops, tau, Pker):
        self.n, self.ops = n, ops
        V = np.kron(np.eye(D), np.diag(np.exp(-1j * Theta / n * M_OF)))
        self.V = V
        self.tau = V @ tau @ V.conj().T
        self.Pk = V @ Pker @ V.conj().T
        self.kdim = round(np.trace(Pker).real)
        self.G = herm_basis(self.tau)
        self.h0 = self.tau / np.trace(self.tau @ self.tau).real
        self.v0 = None
        self.cache = {}

    def ceiling(self):
        hc = np.eye(9) - 9 / self.kdim * self.Pk
        return np.array([np.trace(hc @ G).real for G in self.G])


CASOS = {  # (a, b1, b2, c); los aleatorios con semilla fija
    "AKLT": (np.sqrt(2 / 3), -np.sqrt(1 / 3), np.sqrt(1 / 3), -np.sqrt(2 / 3)),
}


def aleatorios(k, lam_range=(0.3, 0.6), seed=7):
    rng = np.random.default_rng(seed)
    out = {}
    while len(out) < k:
        z = rng.standard_normal(4) + 1j * rng.standard_normal(4)
        A, L, mods, E = normalize(tensors(*z))
        lam = mods[1]
        if lam_range[0] < lam < lam_range[1] and abs(z[1] + z[2]) > 0.3:
            out[f"aleatorio{len(out) + 1}"] = tuple(z)
    return out


if __name__ == "__main__":
    nmax = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 12
    casos = dict(CASOS) if "--aklt" in sys.argv else {}
    casos.update(aleatorios(12))
    max_validos = 3
    thetas = tuple(np.linspace(0, np.pi, 5))
    ns = list(range(6, nmax + 1))
    rows = []
    validos = 0
    for name, z in casos.items():
        if validos >= max_validos:
            break
        A, L, mods, E = normalize(tensors(*z))
        lam = mods[1]
        J = jordan_size(E, lam)
        normal = np.allclose(E @ E.conj().T, E.conj().T @ E, atol=1e-10)
        sigma = marginal(A, L)
        w, U = np.linalg.eigh(sigma)
        ker = U[:, w < 1e-12]
        Pker = ker @ ker.conj().T
        tau = sigma - np.eye(9) / 9
        print(f"\n{name}: z = {np.round(z, 3)}, |mu| = {np.round(mods, 4)}, lambda = {lam:.5f}, J = {J}, "
              f"canal normal: {normal}, rango sigma = {9 - ker.shape[1]}", flush=True)
        check(abs(np.trace(sigma) - 1) < 1e-12 and np.all(w > -1e-12), f"{name}: sigma es un estado")
        r1, r2 = np.trace(sigma.reshape(3, 3, 3, 3), axis1=0, axis2=2), np.trace(sigma.reshape(3, 3, 3, 3), axis1=1, axis2=3)
        check(np.allclose(r1, r2, atol=1e-12), f"{name}: tr_1 sigma = tr_2 sigma")
        print("    n    1-p(0)      1-p(pi)     (1-p(0))/lam^n   (1-p(0))/lam^2n   (1-p(pi))/lam^2n   "
              "max(p_sup-p_inf)", flush=True)
        for n in ns:
            ops = [sector_ops(n, M) for M in (0, 1, 2, -1, -2)]
            h_prev, T_prev, ps, cert, valido = None, None, [], 0.0, True
            for T in thetas:
                d = MPSDual(n, T, ops, tau, Pker)
                c0 = d.ceiling() if h_prev is None else d.rotated(h_prev, T_prev, T)
                try:
                    r = d.solve(c0)
                except Exception as e:  # ARPACK sin convergencia: punto degenerado
                    print(f"      n={n}, Theta={T / np.pi:.2f} pi: {type(e).__name__}; caso descartado")
                    valido = False
                    break
                h_prev, T_prev = d.h(r["c"]), T
                sep = max(d.top(r["c"], k=1, ops=ops[k])[0][0] / n for k in range(1, 5)) - r["p_sup"]
                if sep >= -1e-9:
                    # El máximo de H_h sale del sector M = 0: la optimización en M = 0 no
                    # es el dual completo y el punto no está certificado.
                    print(f"      n={n}, Theta={T / np.pi:.2f} pi: máximo fuera de M = 0 ({sep:+.1e}); caso descartado")
                    valido = False
                    break
                ps.append(r["p_sup"])
                cert = max(cert, r["p_sup"] - r["p_inf"])
                rows.append(dict(caso=name, lam=lam, J=J, n=n, Theta=T, p_sup=r["p_sup"], p_inf=r["p_inf"],
                                 gap=r["gap"], sep_M=sep))
            if not valido:
                rows = [x for x in rows if x["caso"] != name]   # el CSV solo lleva casos válidos
                break
            ps = np.array(ps)
            print(f"  {n:3d}  {1 - ps[0]:.3e}   {1 - ps[-1]:.3e}   {(1 - ps[0]) / lam ** n:10.3e}     "
                  f"{(1 - ps[0]) / lam ** (2 * n):10.3e}     {(1 - ps[-1]) / lam ** (2 * n):10.3e}       {cert:.1e}",
                  flush=True)
            if cert > 1e-8:
                print(f"      (n={n} sin certificar a 1e-8: se excluye del análisis)")
        else:
            validos += 1
            continue
    out = ROOT / "data" / "mps_aleatorio.csv"
    with open(out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"escrito {out}")
    sys.exit(0 if aklt_lanczos.OK else 1)
