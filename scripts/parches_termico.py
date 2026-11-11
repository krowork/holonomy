"""Artículo III (parches): el estado de máxima entropía como estado térmico. Ley de área, límite termodinámico y umbrales.

1. Ley de área de la información mutua (Wolf-Verstraete-Hastings-Cirac). Con rho_G = exp(K)/Z, K = sum_e x_e . O_e, y un corte
   A|B, la minimalidad de la energía libre da I(A:B) <= tr[K_d (rho_G - rho_A x rho_B)], con K_d los términos de las aristas
   de corte. Como las marginales de un parche son I/2, tr[K_d rho_A x rho_B] = 0, y
       I(A:B) <= tr(K_d rho_G) = sum_{e en el corte} x_e . t_e.
   En un nervio transitivo en aristas sin datos, K = -b sum_e s_i.s_j, con b = beta/(3|E|), y la cota es 3 b v |corte|.
   Se comprueba en los nervios y cortes de data/parches_nervios_mitades.csv.
2. Temperatura por arista: b = -(1/(3|E|)) dS_G/dv, por diferencias finitas, en los nervios transitivos en aristas.
3. Límite termodinámico en el toro triangular L x L sin datos: S_G/N = min_b [ln Z_L(b)/N - 9 b v], con Z_L la función de
   partición de H = sum s_i.s_j. Espectro completo por sectores de S_z para L = 3 (control frente a parches_nervios) y 4.
4. Umbrales de alta temperatura: b(v) a v pequeño frente a b_sep = 1/(200 (6 Delta - 4)) (Bakshi-Liu-Moitra-Tang, Teorema 1.5,
   términos de Pauli XX, YY, ZZ de norma b, localidad 2, grado del grafo dual 6 Delta - 4) y b_CMI = 1/(48 e^3 Delta)
   (Kuwahara-Kato-Brandao, Teorema 1, normalización sum_{X ni i} ||h_X|| = 3 b Delta, k = 2).

Escribe data/parches_termico_{area,limite,umbral}.csv.

Uso: (ulimit -v 16000000; OMP_NUM_THREADS=8 python scripts/parches_termico.py)
"""
import _common  # noqa: F401

import csv
import itertools

import numpy as np
import scipy.sparse as sp

from holonomy.report import write_csv, DATA
from parches_nervios import (toro, poliedro, operador_K, gibbs_denso, reduce_a, entropia)
from sphere_entropia import Gibbs

FALLOS = []


def ok(cond, msg):
    print(("OK   " if cond else "FALLO ") + msg, flush=True)
    if not cond:
        FALLOS.append(msg)


def casos():
    out = [(f"{nom} {caso}", *poliedro(nom, caso)) for nom in ("tetraedro", "octaedro", "icosaedro")
           for caso in ("trivial", "curvado")]
    out += [(f"toro {L1}x{L2}", *toro(L1, L2)) for L1, L2 in ((3, 3), (3, 4))]
    return out


TRANSITIVOS = ("tetraedro trivial", "octaedro trivial", "icosaedro trivial", "toro 3x3")


def parte_area():
    print("\n1-2. Ley de área I(A:B) <= sum_corte x_e.t_e y temperatura por arista")
    with open(DATA / "parches_nervios_mitades.csv") as fh:
        tabla = list(csv.DictReader(fh))
    filas = []
    for nombre, g, ph, orbs, mitad in casos():
        G = Gibbs(g, ph, orbs)
        corte_e = [m for m, (i, j) in enumerate(g.edges) if (i in mitad) != (j in mitad)]
        y = None
        for r in [r for r in tabla if r["grafo"] == nombre]:
            v = float(r["v"])
            y, S, res = G.resuelve(v, y0=y)
            a, bb, c = G.dual.edge_coefficients(y)
            rho = gibbs_denso(operador_K(g, ph, (a, bb, c)))
            m = np.zeros(len(g.edges))
            m[corte_e] = 1
            Kd = operador_K(g, ph, (a * m, bb * m, c * m))
            cota = float(np.real(np.vdot(Kd.conj().T, rho)))  # tr(K_d rho)
            rA = reduce_a(rho, g.n, mitad)
            B = [q for q in range(g.n) if q not in mitad]
            rB = reduce_a(rho, g.n, B)
            I = entropia(rA) + entropia(rB) - entropia(rho)
            # tr(K_d rho_A x rho_B), en el orden de qubits A + B
            perm = list(mitad) + B
            Kp = Kd.reshape([2] * (2 * g.n)).transpose(perm + [g.n + q for q in perm]).reshape(2**g.n, 2**g.n)
            prod = float(np.real(np.vdot(Kp.conj().T, np.kron(rA, rB))))
            fila = [nombre, g.n, len(corte_e), r["r"], v, I, float(r["I"]), cota, prod]
            linea = (f"  {nombre:18s} r = {r['r']:4s}: I = {I:.5f} (tabla {float(r['I']):.5f}), cota {cota:.5f}, "
                     f"I/cota = {I / cota:.3f}, tr(K_d rho_A x rho_B) = {prod:.1e}")
            if nombre in TRANSITIVOS:
                bvals = np.concatenate([a, c])
                b = bvals.mean()
                h = 1e-5
                Sp = G.resuelve(v + h, y0=y)[1]
                Sm = G.resuelve(v - h, y0=y)[1]
                b_fd = -(Sp - Sm) / (2 * h) / (3 * len(g.edges))
                fila += [b, b_fd]
                linea += f"; b = {b:.6f}, -(dS/dv)/(3|E|) = {b_fd:.6f}"
                ok(np.ptp(bvals) < 1e-7 and abs(b - b_fd) < 1e-5 * max(1, b)
                   and abs(cota - 3 * b * v * len(corte_e)) < 1e-8,
                   f"{nombre} r={r['r']}: K = -b sum s.s, b = -(dS/dv)/(3|E|), cota = 3 b v |corte|")
            else:
                fila += ["", ""]
            print(linea, flush=True)
            filas.append(fila)
            ok(abs(I - float(r["I"])) < 1e-6 and I <= cota + 1e-9 and abs(prod) < 1e-9,
               f"{nombre} r={r['r']}: I reproduce la tabla y cumple la cota")
    write_csv("parches_termico_area.csv",
              ("grafo", "N", "aristas_corte", "r", "v", "I", "I_tabla", "cota", "K_d_producto", "b", "b_dSdv"), filas)


# ---------------------------------------------------------------------------
# Heisenberg en el toro triangular, por sectores de S_z
# ---------------------------------------------------------------------------

def espectro_heisenberg(n, edges):
    """Autovalores de H = sum_e s_i.s_j (matrices de Pauli) con sus multiplicidades, por sectores de S_z."""
    out = []
    for k in range(n // 2 + 1):  # k espines abajo; k y n - k dan el mismo espectro
        estados = np.array([sum(1 << q for q in c) for c in itertools.combinations(range(n), k)], dtype=np.int64)
        pos = {s: m for m, s in enumerate(estados.tolist())}
        diag = np.zeros(len(estados))
        fil, col = [], []
        for i, j in edges:
            bi, bj = (estados >> i) & 1, (estados >> j) & 1
            diag += np.where(bi == bj, 1.0, -1.0)
            for m in np.nonzero(bi != bj)[0]:
                fil.append(m)
                col.append(pos[int(estados[m] ^ ((1 << i) | (1 << j)))])
        H = sp.coo_matrix((np.full(len(fil), 2.0), (fil, col)), shape=(len(estados),) * 2).toarray()
        H[np.diag_indices_from(H)] += diag
        w = np.linalg.eigvalsh(H)
        out.append((w, 1 if 2 * k == n else 2))
        print(f"    sector k = {k}: dim {len(estados)}", flush=True)
    return out


def S_por_sitio(esp, n, v):
    """min_b [ln Z(b)/N - 9 b v] por Newton en b (convexo); devuelve s y b."""
    E = np.concatenate([w for w, _ in esp])
    g = np.concatenate([np.full(len(w), m) for w, m in esp])
    b = v
    for _ in range(200):
        x = -b * E
        mx = x.max()
        p = g * np.exp(x - mx)
        Z = p.sum()
        p /= Z
        m1 = (p * E).sum()
        var = (p * E**2).sum() - m1**2
        grad = -m1 / n - 9 * v
        b_new = b - grad / (var / n)
        if abs(b_new - b) < 1e-14:
            break
        b = b_new
    x = -b * E
    lnZ = x.max() + np.log((g * np.exp(x - x.max())).sum())
    return lnZ / n - 9 * b * v, b


def parte_limite():
    print("\n3. Toro triangular L x L: S_G/N = min_b [ln Z_L(b)/N - 9 b v]")
    with open(DATA / "parches_nervios_entropia.csv") as fh:
        ref = {round(float(r["v"]), 6): float(r["S"]) / 9 for r in csv.DictReader(fh) if r["grafo"] == "toro 3x3"}
    with open(DATA / "parches_nervios_vmax.csv") as fh:
        vmax = {r["grafo"]: float(r["v_max"]) for r in csv.DictReader(fh)}
    vmax.setdefault("toro 4x4", 0.2376532)  # tabla del nervio plano (dual de órbitas, --grande de parches_nervios), 7 cifras
    filas = []
    for L in (3, 4):
        g = toro(L, L)[0]
        print(f"  L = {L}:", flush=True)
        esp = espectro_heisenberg(g.n, g.edges)
        e0 = min(w.min() for w, _ in esp) / g.n
        vm = 4 / 9 * abs(e0) / 4  # e_0 en unidades de S = s/2: H_S = H/4
        ok(abs(vm - vmax[f"toro {L}x{L}"]) < 1e-7, f"L = {L}: v_max = |e_0|/9 (Pauli) = {vm:.7f}")
        for v in (0.05, 0.1, 0.15, 0.2):
            if v >= vm:
                continue
            s, b = S_por_sitio(esp, g.n, v)
            filas.append([L, g.n, v, s, b])
            print(f"    v = {v:.2f}: s = {s:.6f}, b = {b:.6f}", flush=True)
            if L == 3 and round(v, 6) in ref:
                ok(abs(s - ref[round(v, 6)]) < 1e-9, f"L = 3, v = {v}: igual que parches_nervios ({ref[round(v, 6)]:.6f})")
    write_csv("parches_termico_limite.csv", ("L", "N", "v", "s", "b"), filas)


def parte_umbral():
    print("\n4. b(v) a v pequeño frente a los umbrales de separabilidad y de CMI")
    filas = []
    for nombre, g, ph, orbs, _ in casos():
        if nombre not in TRANSITIVOS:
            continue
        deg = 2 * len(g.edges) // g.n
        b_sep = 1 / (200 * (6 * deg - 4))
        b_cmi = 1 / (48 * np.e**3 * deg)
        G = Gibbs(g, ph, orbs)
        y = None
        for v in (1e-4, 1e-3, 1e-2):
            y, S, res = G.resuelve(v, y0=y)
            a, _, c = G.dual.edge_coefficients(y)
            b = a.mean()
            filas.append([nombre, deg, v, b, b_sep, b_cmi])
            print(f"  {nombre:18s} grado {deg}: v = {v:.0e}, b = {b:.6e}, b/v = {b / v:.4f}; b_sep = {b_sep:.2e}, "
                  f"b_CMI = {b_cmi:.2e}", flush=True)
        ok(abs(filas[-3][3] / 1e-4 - 1) < 1e-2, f"{nombre}: b = v + O(v^2)")
    write_csv("parches_termico_umbral.csv", ("grafo", "grado", "v", "b", "b_sep", "b_CMI"), filas)


def main():
    parte_area()
    parte_umbral()
    parte_limite()
    print(f"\n{'TODO OK' if not FALLOS else f'{len(FALLOS)} FALLOS'}")
    raise SystemExit(bool(FALLOS))


if __name__ == "__main__":
    main()
