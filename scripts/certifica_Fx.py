"""Certificación de la hipótesis (F_x) del Lema del testigo (sensibilidad_celda3.tex, datos de otro anillo).

(F_x): el estado de momento 0 de H_x = H_n(Theta) + sum_a x*_a O_a(Theta) es el fundamental. Fuera del multiplete
de los tres patrones basta Weyl (hueco 1, perturbación << 1/2). Dentro del multiplete se usa el complemento de Schur
sobre P = span(patrones):
    H_eff(E) = P H' P + sum_{j >= 2} P V (R(E) V)^{j-1} P,   H' = H0 + V_d (diagonal),  V = eps T + V_o,
    R(E) = Q (E - H')^{-1} Q   (diagonal),
con V_d la parte diagonal de sum x_a O_a y V_o la no diagonal. H_x conmuta con la traslación, así que H_eff es
circulante: autovalores lambda_k(E) = c0 + c1 w^k + c2 w^{2k}, k = 0 el de momento 0, y E_k es la única raíz de
E - lambda_k(E) en la ventana. Si lambda_k(E) > lambda_0(E) en toda la ventana, E_k > E_0.
Término principal: los caminos de orden D = 2n/3 con x = 0 y E = E0, calculados exactamente (sus coeficientes por
eps^D). Error: mayorante entrada a entrada,
    |c_m - c_m^{(D)}| <= sum_j [A (|R_delta| A)^{j-1}]_{c+m, c} - [|T| (|R_0| |T|)^{D-1}]_{c+m, c} eps^D,
A = eps |T| + sum_o |x_o| |O_o|, |R_delta|_s = 1 / (d_s - delta), delta = 2 |V_d| + |V| (desplazamiento de E y de
H' en la ventana), más la cola j > K por normas. Las cotas de x* salen de Kantorovich (kantorovich_datos.py):
|u_i| <= 2 eta W_i, y = S^{-1} u, x = B(eps) y.
Uso: python scripts/certifica_Fx.py serie_n.npz datos.npz [inf] [eps_a Rh Rm Rs wm ws]
"""
import os
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_argv, sys.argv = sys.argv, sys.argv[:1]
import kantorovich_series as KS  # noqa: E402
import kantorovich_datos as KD  # noqa: E402
import newton_graduado as NG  # noqa: E402
import serie_respuesta as SR  # noqa: E402
sys.argv = _argv


class Multiplete:
    def __init__(self, n, Ks=12):
        self.n, self.D = n, 2 * n // 3
        self.K = self.D + Ks
        H0, T, Os, P = SR.operadores(n, 0.0)
        self.H0, self.P = H0, P
        E0 = H0[P[0]]
        self.d = H0 - E0                                   # >= 1 fuera de los patrones
        self.Q = np.ones(H0.size, bool); self.Q[P] = False
        assert self.d[self.Q].min() >= 1 - 1e-12
        self.aT = abs(T).tocsr()
        self.diag = []; self.aO = []
        for O in Os:
            off = O - sp.diags(O.diagonal())
            self.diag.append(np.abs(O.diagonal()))
            self.aO.append(abs(off).tocsr())
        self.rowT = np.asarray(self.aT.sum(1)).ravel().max()
        self.rowO = np.array([np.asarray(a.sum(1)).ravel().max() if a.nnz else 0.0 for a in self.aO])
        self.dO = np.array([d.max() for d in self.diag])
        # término principal: P T (R0 T)^{D-1} P, por Theta
        self.lead = {}
        R0 = np.where(self.Q, -1 / np.where(self.Q, self.d, 1), 0.0)
        for th in np.linspace(0, 2 * np.pi, 9)[:-1]:
            Tt = SR.operadores(n, th)[1]
            M = np.zeros((3, 3), complex)
            for c in range(3):
                v = np.zeros(H0.size, complex); v[P[c]] = 1
                for _ in range(self.D - 1):
                    v = R0 * (Tt @ v)
                M[:, c] = (Tt @ v)[P]
            self.lead[th] = M
        # mayorante del principal: |T| (|R0| |T|)^{D-1}
        aR0 = np.where(self.Q, 1 / np.where(self.Q, self.d, 1), 0.0)
        Ml = np.zeros((3, 3))
        for c in range(3):
            v = np.zeros(H0.size); v[P[c]] = 1
            for _ in range(self.D - 1):
                v = aR0 * (self.aT @ v)
            Ml[:, c] = (self.aT @ v)[P]
        self.lead_abs = Ml

    def huecos_principales(self):
        """min_k (lambda_k - lambda_0) del término principal (coeficiente de eps^D) sobre los Theta de la rejilla."""
        w = np.exp(2j * np.pi / 3)
        out = {}
        for th, M in self.lead.items():
            # circulante: columna c -> c+m; lambda_k = sum_m M[(c+m)%3, c] w^{mk}; el k = 0 es el simétrico
            # (con la permutación de traslación que sea, el de momento 0 es la suma de los patrones)
            ev = np.linalg.eigvals(M)
            v0 = M.sum(0)[0]
            others = sorted(ev, key=lambda z: abs(z - v0))[1:]
            out[th] = (np.real(v0), [np.real(z) for z in others], M)
        return out

    def error(self, eps, xabs):
        """Cota de |c_m - principal| (m = 1, 2) para |eps'| <= eps, |x_a| <= xabs, toda E de la ventana."""
        A = eps * self.aT
        for a, xa in enumerate(xabs):
            if self.aO[a].nnz and xa > 0:
                A = A + xa * self.aO[a]
        Vd = float(np.dot(xabs, self.dO)) * 1.0              # |V_d| <= sum |x_a| max|diag O_a|
        nA = eps * self.rowT + float(np.dot(xabs, self.rowO))  # norma (suma de fila) de A >= |V_o + eps T|
        delta = 2 * Vd + nA
        if delta >= 0.5 or nA / (1 - delta) >= 0.5:
            return np.inf, delta
        aR = np.where(self.Q, 1 / np.where(self.Q, self.d - delta, 1), 0.0)
        tot = np.zeros((3, 3))
        for c in range(3):
            v = np.zeros(self.H0.size); v[self.P[c]] = 1
            for j in range(1, self.K + 1):
                w = A @ v
                tot[:, c] += w[self.P]
                v = aR * w
        r = nA / (1 - delta)
        cola = (1 - delta) * r ** (self.K + 1) / (1 - r)
        err = tot - self.lead_abs * eps ** self.D + cola
        err[np.arange(3), np.arange(3)] = 0.0               # la diagonal c0 es común a todos los k
        return err.max(), delta


def x_cotas(c, el, er, wv, B0, B1):
    W = KS.BLOQ.choose([1.0, wv[1], wv[2]])
    u = 2 * c['eta'] * W
    y = u / el ** KS.EXPO.astype(float)
    return (np.abs(B0) + er * np.abs(B1)) @ y


def rejilla(S, P, D, n):
    """La misma rejilla gruesa de parámetros del polidisco que kantorovich_datos.py."""
    base = {9: (0.03, 1e-4, 4e-3, 0.1, 1, 1), 12: (0.02, 2e-5, 1e-3, 0.1, 0.3, 0.3)}[n]
    best = (0.0, None)
    for fa in (0.5, 0.75, 1, 1.25):
        for fh in (0.5, 1, 2):
            for fm in (0.5, 1, 2):
                for fs in (0.5, 1, 1.5):
                    for fw in (0.3, 1, 3):
                        par = (base[0] * fa, base[1] * fh, base[2] * fm, base[3] * fs, base[4] * fw, base[5] * fw)
                        e_ok = 0.0
                        for e in np.geomspace(1e-9, 2e-2, 60):
                            c = KD.cotas(S, P, D, e, e * 1.02, par[0], par[1:4], np.array([1, par[4], par[5]]))
                            if c is None or c['h'] > 0.5:
                                break
                            e_ok = e
                        if e_ok > best[0]:
                            best = (e_ok, par)
    return best[1]


if __name__ == "__main__":
    S = KS.carga(sys.argv[1])
    P = KS.preparar(S)
    inf = 'inf' in sys.argv[3:]
    D = KD.datos(sys.argv[2], S['K'], S['n'], inf=inf)
    n = S['n']
    nums = [float(x) for x in sys.argv[3:] if x != 'inf']
    par = tuple(nums) if nums else rejilla(S, P, D, n)
    eps_a, R, wv = par[0], par[1:4], np.array([1, par[4], par[5]])
    B0 = NG.base_explicita(0.0); B1 = NG.base_explicita(1.0) - B0
    M = Multiplete(n)
    print(f"n={n}, D={M.D}, datos {'inf' if inf else D['N']}, par={par}")
    print(f"  |T| fila max = {M.rowT:.2f};  operadores no diagonales: {[a for a in range(len(M.aO)) if M.aO[a].nnz]}")
    hp = M.huecos_principales()
    gap_lead = np.inf
    for th, (l0, ls, Mx) in hp.items():
        g = min(ls) - l0
        gap_lead = min(gap_lead, g)
        print(f"  Theta={th:.3f}: principal lambda_0 = {l0:.6g}, otros {ls[0]:.6g} {ls[1]:.6g}; "
              f"hueco = {g:.6g} eps^{M.D};  |principal| vs mayorante: {np.abs(Mx).max():.6g} / {M.lead_abs.max():.6g}")
    if gap_lead <= 0:
        raise SystemExit("  el término principal no deja el momento 0 abajo")
    # barrido en intervalos [el, el*paso]
    el, paso, ok_hasta, primero_malo = 1e-30, 1.01, None, None
    peor = 0.0
    while el < 0.1:
        er = el * paso
        c = KD.cotas(S, P, D, el, er, eps_a, R, wv)
        if c is None or c['h'] > 0.5:
            print(f"  Kantorovich deja de valer en {el:.3e}")
            break
        xa = x_cotas(c, el, er, wv, B0, B1)
        err, delta = M.error(er, xa)
        # lambda_k - lambda_0 >= gap_lead el^D - |w^k - 1| (|err_1| + |err_2|) >= gap_lead el^D - 2 sqrt3 err
        marg = 2 * np.sqrt(3) * err / (gap_lead * el ** M.D)
        peor = max(peor, marg)
        if marg < 1:
            ok_hasta = er
        elif primero_malo is None:
            primero_malo = (el, marg, xa.max(), delta)
        el = er
    print(f"  (F_x) certificada en [1e-30, {ok_hasta:.3e}]" if ok_hasta and primero_malo is None
          else f"  (F_x): primer fallo en eps = {primero_malo[0]:.3e} (cociente {primero_malo[1]:.2f}, "
               f"|x| <= {primero_malo[2]:.1e}, delta = {primero_malo[3]:.1e}); vale hasta {ok_hasta}")
    print(f"  peor cociente error/hueco en el rango: {peor:.3e}")
