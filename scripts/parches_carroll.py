"""Borrador «Parches causales»: comprobaciones numéricas.

1. Criterio de arista con un boost de Carroll. Con T = -v M y
   M = R_z(phi) B(b) D(lam), B(b) el boost (e_a -> e_a + b_a l, l = e_z) y
   D(lam) = diag(1, 1, lam), el estado de dos qubits
   rho = (I + sum_ab T_ab s_a s_b)/4 es positivo si y solo si
   v (s1 + s2 - s3) <= 1, con s1 >= s2 >= s3 los valores singulares de M
   (det M > 0). Con lam = 1 eso es v <= 1/(1 + |b|).
2. Los valores singulares de T son invariantes bajo el gauge O_i T O_j^T:
   el boost no se puede quitar con unitarios locales.
3. Los nervios: el octaedro y el icosaedro son los nervios de los
   recubrimientos de la esfera por casquetes centrados en sus vértices, y el
   cubo no es el nervio de ninguno con casquetes iguales (no tiene triángulos).
"""
import itertools

import numpy as np

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1.0 + 0j, -1.0])
S = [X, Y, Z]


def rz(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def boost(b1, b2):
    # e_a -> e_a + b_a l, con l = e_z: la fila z lleva b
    return np.array([[1, 0, 0], [0, 1, 0], [b1, b2, 1.0]])


def rho_of(T):
    r = np.eye(4, dtype=complex)
    for a in range(3):
        for c in range(3):
            r += T[a, c] * np.kron(S[a], S[c])
    return r / 4


def v_arista(M):
    s = np.linalg.svd(M, compute_uv=False)
    assert np.linalg.det(M) > 0
    return 1 / (s[0] + s[1] - s[2])


def comprobar_criterio(rng, n=2000):
    peor = 0.0
    for _ in range(n):
        phi = rng.uniform(-np.pi, np.pi)
        b = rng.normal(size=2) * rng.uniform(0, 3)
        lam = np.exp(rng.uniform(-1.5, 1.5)) if rng.random() < 0.5 else 1.0
        M = rz(phi) @ boost(*b) @ np.diag([1, 1, lam])
        v0 = v_arista(M)
        if lam == 1.0:
            peor = max(peor, abs(v0 - 1 / (1 + np.hypot(*b))))
        lo = np.linalg.eigvalsh(rho_of(-(v0 * (1 - 1e-7)) * M)).min()
        hi = np.linalg.eigvalsh(rho_of(-(v0 * (1 + 1e-7)) * M)).min()
        assert lo > -1e-12 and hi < 0, (phi, b, lam, lo, hi)
    return peor


def comprobar_gauge(rng):
    M = boost(0.7, -0.4)
    s0 = np.linalg.svd(M, compute_uv=False)
    for _ in range(100):
        Oi, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        Oj, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        assert np.allclose(np.linalg.svd(Oi @ M @ Oj.T, compute_uv=False), s0)
    return s0


def vertices(nombre):
    f = (1 + np.sqrt(5)) / 2
    if nombre == "octaedro":
        p = [v for i in range(3) for v in (np.eye(3)[i], -np.eye(3)[i])]
    elif nombre == "icosaedro":
        p = []
        for s1, s2 in itertools.product((1, -1), repeat=2):
            for k in range(3):
                q = np.roll([0.0, s1 * 1.0, s2 * f], k)
                p.append(q)
    elif nombre == "cubo":
        p = [np.array(s, float) for s in itertools.product((1, -1), repeat=3)]
    return [x / np.linalg.norm(x) for x in p]


def nervio(puntos, radio, muestras):
    """Aristas y triángulos del nervio de los casquetes abiertos de radio
    angular `radio`, estimados con puntos de la esfera."""
    dentro = np.arccos(np.clip(muestras @ np.array(puntos).T, -1, 1)) < radio
    cubierta = dentro.any(axis=1).all()
    orden = dentro.sum(axis=1).max()
    aristas = {(i, j) for i, j in itertools.combinations(range(len(puntos)), 2)
               if (dentro[:, i] & dentro[:, j]).any()}
    tri = {t for t in itertools.combinations(range(len(puntos)), 3)
           if (dentro[:, t[0]] & dentro[:, t[1]] & dentro[:, t[2]]).any()}
    return cubierta, orden, aristas, tri


def esqueleto(puntos):
    d = np.array([[np.arccos(np.clip(a @ b, -1, 1)) for b in puntos] for a in puntos])
    m = np.min(d[d > 1e-9])
    return {(i, j) for i, j in itertools.combinations(range(len(puntos)), 2)
            if abs(d[i, j] - m) < 1e-9}


def main():
    rng = np.random.default_rng(0)
    peor = comprobar_criterio(rng)
    print(f"criterio de arista: 2000 casos, |v - 1/(1+|b|)| <= {peor:.1e}")
    print("valores singulares del boost (0,7; -0,4):", np.round(comprobar_gauge(rng), 6))
    for b in (0.0, 0.5, 1.0, 2.0):
        print(f"  |b| = {b}: v_arista = {1/(1+b):.6f}")
    muestras = rng.normal(size=(400000, 3))
    muestras /= np.linalg.norm(muestras, axis=1, keepdims=True)
    for nombre, radio in (("octaedro", 1.0), ("icosaedro", 0.70)):
        p = vertices(nombre)
        cub, orden, ar, tri = nervio(p, radio, muestras)
        print(f"{nombre}: radio {radio}, cubre {cub}, orden {orden}, "
              f"nervio = esqueleto: {ar == esqueleto(p)}, triángulos {len(tri)}")
    # el cubo: casquetes iguales que cubren la esfera
    p = vertices("cubo")
    for radio in (0.96, 1.0, 1.1):
        cub, orden, ar, tri = nervio(p, radio, muestras)
        print(f"cubo: radio {radio}, cubre {cub}, aristas {len(ar)}, triángulos {len(tri)}, "
              f"nervio = esqueleto: {ar == esqueleto(p)}")


if __name__ == "__main__":
    main()
