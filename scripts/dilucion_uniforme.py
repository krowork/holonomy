"""Dilución uniforme (Lema dilucionti de docs/robustez_mps.tex).

Para Delta hermítico de dos sitios, de traza nula, con tr_1 Delta = tr_2 Delta = delta, y
||Delta||_1 <= 1/(19 d^2), construye un estado del anillo de n >= 3 sitios con marginal
I/d^2 + Delta en todas las aristas, con un radio que no depende de n:

  omega = (omega_C + omega_delta)/2,
  omega_delta = (I/d + 2 delta)^{(x) n},
  omega_C = media sobre traslaciones de pares (I/d^2 + alpha Ct) en (1,2),(3,4),...
            [+ un sitio I/d si n es impar], Ct = C - 2 delta (x) delta,
  alpha = 4 (n par), 4n/(n-1) (n impar).

Comprueba con Delta aleatorios en el borde del radio que omega es semidefinido positivo y que
sus marginales de arista son exactamente I/d^2 + Delta.
"""
import sys

import numpy as np

rng = np.random.default_rng(1)


def kron(*ops):
    out = np.eye(1)
    for x in ops:
        out = np.kron(out, x)
    return out


def ptrace(rho, keep, n, d):
    """Traza parcial dejando los sitios `keep` (en orden creciente)."""
    r = rho.reshape([d] * (2 * n))
    for i in sorted((i for i in range(n) if i not in keep), reverse=True):
        m = r.ndim // 2
        r = np.trace(r, axis1=i, axis2=i + m)
    k = len(keep)
    return r.reshape(d**k, d**k)


def permutar(rho, perm, n, d):
    """Lleva el sitio i al sitio perm[i]."""
    inv = np.argsort(perm)
    r = rho.reshape([d] * (2 * n))
    return r.transpose(list(inv) + [n + i for i in inv]).reshape(d**n, d**n)


def delta_aleatorio(d, radio):
    I = np.eye(d)
    H = rng.normal(size=(d * d, d * d)) + 1j * rng.normal(size=(d * d, d * d))
    H = H + H.conj().T
    H -= np.trace(H) / d**2 * np.eye(d * d)
    R = H.reshape(d, d, d, d)
    a = np.einsum("abad->bd", R)  # reducción al segundo sitio
    b = np.einsum("abcb->ac", R)  # reducción al primer sitio
    m = (a + b) / 2
    H = H - np.kron(b - m, I / d) - np.kron(I / d, a - m)
    return H / np.abs(np.linalg.eigvalsh(H)).sum() * radio


def construir(D, d, n):
    I = np.eye(d)
    R = D.reshape(d, d, d, d)
    de = np.einsum("abad->bd", R)
    assert np.allclose(de, np.einsum("abcb->ac", R))
    C = D - np.kron(de, I / d) - np.kron(I / d, de)
    Ct = C - 2 * np.kron(de, de)
    om_d = kron(*[I / d + 2 * de] * n)
    alpha = 4.0 if n % 2 == 0 else 4 * n / (n - 1)
    bloques = [np.eye(d * d) / d**2 + alpha * Ct] * (n // 2) + ([I / d] if n % 2 else [])
    base = kron(*bloques)
    om_C = sum(permutar(base, [(i + s) % n for i in range(n)], n, d) for s in range(n)) / n
    return (om_C + om_d) / 2


def main():
    ok = True
    for d, ns in [(2, range(3, 8)), (3, range(3, 6))]:
        for n in ns:
            D = delta_aleatorio(d, 1 / (19 * d * d))
            om = construir(D, d, n)
            obj = np.eye(d * d) / d**2 + D
            sw = lambda X: X.reshape(d, d, d, d).transpose(1, 0, 3, 2).reshape(d * d, d * d)
            err = max(
                np.abs((ptrace(om, [j, j + 1], n, d) if j < n - 1 else sw(ptrace(om, [0, n - 1], n, d))) - obj).max()
                for j in range(n)
            )
            emin = np.linalg.eigvalsh(om).min()
            print(f"d={d} n={n}: autovalor mínimo {emin:.2e}, error de marginales {err:.1e}")
            ok &= emin > 0 and err < 1e-12 and abs(np.trace(om) - 1) < 1e-12
    print("correcto" if ok else "FALLA")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
