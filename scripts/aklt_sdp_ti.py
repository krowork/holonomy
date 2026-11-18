"""p(n, Theta) exacto con datos AKLT hasta n = 8, por el dual y un primal restringido.

p(n, Theta) es el mayor p para el que existe un estado del anillo de n espines 1
con marginales de arista I/9 + p tau_Theta, tau_Theta = (1 ⊗ V) tau (1 ⊗ V)^†,
V = exp(-i Theta S^z / n), tau = sigma_AKLT - I/9 (gauge uniforme: el mismo dato
en todas las aristas; scripts/aklt_twist.py lo resuelve con el SDP completo en
n = 4, 5, y el SDP completo no cabe en memoria desde n = 6).

Dual. Para todo h hermítico de dos sitios con tr(h tau_Theta) = 1,

    p(n, Theta) <= lambda_max(H_h) / n - tr(h) / 9,   H_h = sum_j h_{j, j+1},

y hay igualdad en el ínfimo (Slater: I/3^n es interior con p = 0). H_h conmuta con
la traslación y con S^z total, así que es diagonal por bloques (M, momento k), de
tamaño <= 138 en n = 8. Se minimiza la versión suavizada (log-sum-exp a
temperatura 1/beta creciente, L-BFGS) y se evalúa la cota exacta en el h final.

Primal. Por holgura complementaria el óptimo primal vive en el espacio propio
superior del h óptimo. Si ese autovector es único, se pule con Newton la
estacionariedad, que equivale a que su marginal sea exactamente I/9 + p tau: el
autovector es primal factible y las cotas se tocan (así en n = 6, 7, 8). Si está
degenerado, se resuelve el SDP restringido a los autovectores dentro de una
ventana delta de lambda_max, con Clarabel. En ambos casos la marginal se corrige
con el lema de dilución: cota inferior rigurosa salvo redondeo. El resultado es
un intervalo [p_inf, p_sup].

Validación: n = 4, 5 frente al SDP completo (p(4, 0) = 40/41, y los valores de la
tabla de docs/robustez_mps.tex). Uso: python scripts/aklt_sdp_ti.py [n ...]
"""
import sys
import time

import numpy as np

D = 3
M_OF = np.array([1, 0, -1])                  # base m = +1, 0, -1


def aklt_tau():
    Sz = np.diag([1.0, 0.0, -1.0])
    Sp = np.sqrt(2) * np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    Sx, Sy = (Sp + Sp.T) / 2, (Sp - Sp.T) / (2j)
    SS = (np.kron(Sx, Sx) + np.kron(Sy, Sy) + np.kron(Sz, Sz)).real
    I9 = np.eye(9)
    P0 = (SS + I9) @ (SS - I9) / 3
    P1 = -(SS + 2 * I9) @ (SS - I9) / 2
    sigma = P0 / 3 + 2 / 9 * P1
    return sigma - I9 / 9


TAU = aklt_tau()


def sectors(n):
    """Estados (enteros en base 3, sitio 0 = dígito alto) por M total."""
    digits = np.array(np.unravel_index(np.arange(D ** n), [D] * n)).T
    Mtot = M_OF[digits].sum(axis=1)
    return {M: np.where(Mtot == M)[0] for M in range(-n, n + 1)}


def momentum_bases(states, n):
    """Bases ortonormales de momento k (columnas) dentro de un sector."""
    shift = lambda s: s // D + (s % D) * D ** (n - 1)   # sitio j -> j+1
    index = {s: i for i, s in enumerate(states)}
    seen, orbits = set(), []
    for s in states:
        if s in seen:
            continue
        orb, t = [], s
        while t not in orb:
            orb.append(t)
            t = shift(t)
        seen.update(orb)
        orbits.append(orb)
    bases = {}
    for k in range(n):
        cols = []
        for orb in orbits:
            P = len(orb)
            if (k * P) % n:
                continue
            v = np.zeros(len(states), dtype=complex)
            for j, t in enumerate(orb):
                v[index[t]] = np.exp(2j * np.pi * k * j / n)
            cols.append(v / np.linalg.norm(v))
        if cols:
            bases[k] = np.array(cols).T
    return bases, index


def edge_ops(states, index, n, pairs):
    """Para cada (x, y): listas (filas, columnas) de |y><x| en los sitios 0, 1
    dentro del sector, de modo que tr(rho |y><x|) = <x|rho_01|y>."""
    hi, lo = D ** (n - 1), D ** (n - 2)
    loc = (states // lo)                    # = 3 s_0 + s_1
    out = {}
    for x, y in pairs:
        cols = np.where(loc == x)[0]
        x0, x1 = divmod(x, D)
        y0, y1 = divmod(y, D)
        tgt = states[cols] + (y0 - x0) * hi + (y1 - x1) * lo
        out[(x, y)] = (np.array([index[t] for t in tgt], dtype=int), cols)
    return out




def charge_pairs():
    q2 = np.add.outer(M_OF, M_OF).ravel()
    return [(x, y) for x in range(D * D) for y in range(D * D) if q2[x] == q2[y]]


def blocks(n):
    """Para cada bloque (M, k): pila O[a] = B^† |y_a><x_a| B, con (x_a, y_a) los
    pares que conservan la carga, de modo que <x|rho_01|y> = tr(O rho_blk)."""
    pairs = charge_pairs()
    out = []
    for M, st in sectors(n).items():
        bases, index = momentum_bases(st, n)
        ops = edge_ops(st, index, n, pairs)
        for k, B in bases.items():
            O = np.array([B[ops[xy][0]].conj().T @ B[ops[xy][1]] for xy in pairs])
            out.append(((M, k), O))
    return pairs, out


def herm_basis(tau):
    """Base real ortonormal (Hilbert-Schmidt) de los h hermíticos que conservan la
    carga, ortogonales a I y a tau."""
    pairs = charge_pairs()
    G = []
    for x, y in pairs:
        if x == y:
            E = np.zeros((9, 9), complex); E[x, x] = 1; G.append(E)
        elif x < y:
            E = np.zeros((9, 9), complex); E[x, y] = E[y, x] = 1 / np.sqrt(2); G.append(E)
            E = np.zeros((9, 9), complex); E[x, y] = 1j / np.sqrt(2); E[y, x] = -1j / np.sqrt(2); G.append(E)
    vecs = np.array([np.concatenate([g.real.ravel(), g.imag.ravel()]) for g in G]).T
    fix = np.array([np.concatenate([a.real.ravel(), a.imag.ravel()]) for a in (np.eye(9), tau)]).T
    Qf, _ = np.linalg.qr(fix)
    vecs = vecs - Qf @ (Qf.T @ vecs)
    U, S, _ = np.linalg.svd(vecs, full_matrices=False)
    U = U[:, S > 1e-10]
    return [(u[:81] + 1j * u[81:]).reshape(9, 9) for u in U.T]


class Dual:
    def __init__(self, n, Theta, blk):
        self.n = n
        V = np.kron(np.eye(D), np.diag(np.exp(-1j * Theta / n * M_OF)))
        self.tau = V @ TAU @ V.conj().T
        self.pairs, self.blk = blk
        self.G = herm_basis(self.tau)
        self.h0 = self.tau / np.trace(self.tau @ self.tau).real

    def h(self, c):
        return self.h0 + sum(ci * g for ci, g in zip(c, self.G))

    def spectrum(self, c):
        h = self.h(c)
        hv = np.array([h[x, y] for x, y in self.pairs])
        # B^† h_01 B = sum h[x, y] B^† |x><y| B = sum h[x, y] O[(y, x)]
        idx = {xy: a for a, xy in enumerate(self.pairs)}
        perm = [idx[(y, x)] for x, y in self.pairs]
        out = []
        for key, O in self.blk:
            Hb = np.tensordot(hv, O[perm], axes=1)
            w, U = np.linalg.eigh((Hb + Hb.conj().T) / 2)
            out.append((w, U))           # autovalores de H/n = B^† h_01 B
        return h, out

    def upper(self, c):
        h, sp = self.spectrum(c)
        return max(w[-1] for w, _ in sp) - np.trace(h).real / 9

    def smooth(self, c, beta):
        h, sp = self.spectrum(c)
        lmax = max(w[-1] for w, _ in sp)
        Z = sum(np.exp(beta * (w - lmax)).sum() for w, _ in sp)
        f = lmax + np.log(Z) / beta - np.trace(h).real / 9
        rho = np.zeros(len(self.pairs), complex)
        for (w, U), (_, O) in zip(sp, self.blk):
            W = (U * (np.exp(beta * (w - lmax)) / Z)) @ U.conj().T
            rho += np.einsum("aij,ji->a", O, W)
        R = np.zeros((9, 9), complex)
        for (x, y), r in zip(self.pairs, rho):
            R[x, y] = r
        grad = np.array([np.trace((R - np.eye(9) / 9) @ g).real for g in self.G])
        return f, grad

    def minimize(self, betas=(3, 10, 30, 100, 300, 1000, 3000, 10000, 30000)):
        from scipy.optimize import minimize
        c = np.zeros(len(self.G))
        for beta in betas:
            res = minimize(self.smooth, c, args=(beta,), jac=True, method="L-BFGS-B",
                           options=dict(maxiter=3000, gtol=1e-12, ftol=1e-15))
            c = res.x
        return c

    def restricted_primal(self, c, delta):
        """SDP primal en los autovectores con autovalor >= lambda_max - delta."""
        import cvxpy as cp
        h, sp = self.spectrum(c)
        lmax = max(w[-1] for w, _ in sp)
        sub = []
        for (w, U), (_, O) in zip(sp, self.blk):
            sel = w >= lmax - delta
            if sel.any():
                Us = U[:, sel]
                sub.append((Us, np.einsum("ji,ajk,kl->ail", Us.conj(), O, Us)))
        p = cp.Variable()
        X = [cp.Variable((Ot.shape[1],) * 2, hermitian=True) for _, Ot in sub]
        cons = [x >> 0 for x in X]
        lin = [0] * len(self.pairs)
        tr = 0
        for x, (Us, Ot) in zip(X, sub):
            tr = tr + cp.real(cp.trace(x))
            for a in range(len(self.pairs)):
                lin[a] = lin[a] + cp.sum(cp.multiply(Ot[a].T, x))
        # Con pocos vectores el sistema está sobredeterminado y un residuo de
        # 1e-9 lo haría infactible: se penaliza el residuo máximo t, y el lema de
        # dilución lo corrige después.
        I9 = np.eye(9)
        t = cp.Variable(nonneg=True)
        cons.append(tr == 1)
        for a, (xx, yy) in enumerate(self.pairs):
            if xx > yy:
                continue
            res = lin[a] - I9[xx, yy] / 9 - p * self.tau[xx, yy]
            cons += [cp.abs(cp.real(res)) <= t, cp.abs(cp.imag(res)) <= t]
        prob = cp.Problem(cp.Maximize(p - 1e4 * t), cons)
        prob.solve(solver="CLARABEL")
        if p.value is None:
            return None, prob.status, sum(Ot.shape[1] for _, Ot in sub)
        # marginal real de la solución (proyectada sobre los PSD) y dilución
        R = np.zeros((9, 9), complex)
        trace = 0.0
        for x, (Us, Ot) in zip(X, sub):
            w, Q = np.linalg.eigh(x.value)
            Xp = (Q * np.clip(w, 0, None)) @ Q.conj().T
            trace += np.trace(Xp).real
            for a, (xx, yy) in enumerate(self.pairs):
                R[xx, yy] += np.sum(Ot[a].T * Xp)
        R /= trace
        pv = float(p.value)
        eps = np.abs(np.linalg.eigvalsh(R - I9 / 9 - pv * self.tau)).sum()
        r = 1 / (6 * D * D)
        s = self.n * eps / (r + self.n * eps)
        return (1 - s) * pv, prob.status, sum(Ot.shape[1] for _, Ot in sub)


def polish(d, c, iters=30):
    """Newton sobre la estacionariedad cuando el autovector superior es único.

    Si el estado superior psi(c) de H_h es único, f(c) = lambda_max/n - tr h/9 es
    diferenciable y su gradiente es la proyección de rho_01(psi) - I/9 sobre las
    G_k. Gradiente nulo equivale a rho_01(psi) = I/9 + p tau exactamente: psi es
    primal factible con p = f(c). Jacobiano por diferencias finitas en el bloque
    del máximo; las direcciones planas (a ⊗ 1 - 1 ⊗ a) se resuelven por mínimos
    cuadrados. Devuelve (c, p_inf, p_sup, gap) o None si el máximo no es único."""
    I9 = np.eye(9)

    def top(c):
        h, sp = d.spectrum(c)
        tops = sorted(((w[-1], b) for b, (w, _) in enumerate(sp)), reverse=True)
        b = tops[0][1]
        w, U = sp[b]
        second = max(w[-2] if len(w) > 1 else -np.inf, tops[1][0])
        O = d.blk[b][1]
        v = U[:, -1]
        rho = np.einsum("i,aij,j->a", v.conj(), O, v)
        R = np.zeros((9, 9), complex)
        for (x, y), r in zip(d.pairs, rho):
            R[x, y] = r
        g = np.array([np.trace((R - I9 / 9) @ G).real for G in d.G])
        return g, R, w[-1] - second, w[-1] - np.trace(h).real / 9

    g, R, gap, f = top(c)
    if gap < 1e-7:
        return None
    for _ in range(iters):
        if np.abs(g).max() < 1e-13:
            break
        J = np.zeros((len(g), len(c)))
        hstep = 1e-6
        for k in range(len(c)):
            e = np.zeros(len(c)); e[k] = hstep
            J[:, k] = (top(c + e)[0] - top(c - e)[0]) / (2 * hstep)
        dc, *_ = np.linalg.lstsq(J, -g, rcond=1e-10)
        # paso amortiguado: que el residuo baje
        t = 1.0
        while t > 1e-4:
            g2, R2, gap2, f2 = top(c + t * dc)
            if np.abs(g2).max() < np.abs(g).max():
                break
            t /= 2
        c, g, R, gap, f = c + t * dc, g2, R2, gap2, f2
        if gap < 1e-7:
            return None
    pv = np.trace((R - I9 / 9) @ d.tau).real / np.trace(d.tau @ d.tau).real
    eps = np.abs(np.linalg.eigvalsh(R - I9 / 9 - pv * d.tau)).sum()
    r = 1 / (6 * D * D)
    s = d.n * eps / (r + d.n * eps)
    return c, (1 - s) * pv, d.upper(c), gap


def solve(n, Theta, blk):
    d = Dual(n, Theta, blk)
    c = d.minimize()
    up = d.upper(c)
    pol = polish(d, c)
    if pol is not None:
        c2, low, up2, gap = pol
        # solo si convergió: residuo despreciable y cotas que se tocan
        if np.isfinite(up2) and up2 - low < 1e-7 and up2 <= up + 1e-7:
            return low, min(up, up2), f"Newton, gap {gap:.1e}", 1
    low, st, nv = None, "", 0
    best = None
    for delta in (1e-4, 1e-3, 1e-2, 3e-2):
        try:
            low, st, nv = d.restricted_primal(c, delta)
        except Exception:
            continue
        if low is not None and (best is None or low > best[0]):
            best = (low, st, nv)
    low, st, nv = best if best else (None, "fallo", 0)
    return low, up, st, nv


def lower_bound(n, Theta):
    """Cota del teorema (Proposición AKLT)."""
    c = np.cos(Theta)
    f = 4 / 9 * (2 * (5 + c) + np.sqrt(2) * np.sqrt(245 + 44 * c - c * c))
    eps = f / (3 ** n + (-1) ** n * (1 + 2 * c))
    return 1 / (1 + 54 * n * eps)


REF = {(4, 0): 40 / 41, (4, 2): 0.902726, (5, 1): 0.985079, (5, 2): 0.997422}

if __name__ == "__main__":
    ns = [int(a) for a in sys.argv[1:]] or [4, 5, 6, 7, 8]
    print("   n   Theta/pi    p_inf        p_sup       teorema    vectores  estado")
    ok = True
    for n in ns:
        t = time.time()
        blk = blocks(n)
        print(f"      n = {n}: {len(blk[1])} bloques, máximo {max(O.shape[1] for _, O in blk[1])} "
              f"({time.time() - t:.0f} s)", flush=True)
        for i, Theta in enumerate((0.0, np.pi / 2, np.pi)):
            t = time.time()
            low, up, st, nv = solve(n, Theta, blk)
            th = lower_bound(n, Theta)
            lows = f"{low:.10f}" if low is not None else "   ---   "
            print(f"  {n:2d}   {Theta / np.pi:5.2f}     {lows}    {up:.10f}    {th:.5f}    {nv:5d}    "
                  f"{st}  ({time.time() - t:.0f} s)", flush=True)
            ok &= low is not None and th - 1e-6 <= low <= up + 1e-7 and up <= 1 + 1e-7
            if (n, i) in REF:
                ok &= low - 2e-6 <= REF[(n, i)] <= up + 2e-6
                print(f"         referencia (SDP completo): {REF[(n, i)]:.7f}")
    print("OK" if ok else "FALLO")
    sys.exit(0 if ok else 1)
