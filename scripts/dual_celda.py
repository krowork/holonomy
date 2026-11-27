"""Dual exacto del pegado con datos U(1) genéricos en el anillo: p(n, Theta) por minimización convexa (exploración).

Datos: estado de dos sitios sigma (d^2 x d^2), invariante por la U(1) de cargas q, y tau = sigma - I/d^2; en el gauge
uniforme, la arista j lleva (1 x V) tau (1 x V)^dag con V = e^{-i phi Q}, phi = Theta/n. Por dualidad fuerte
[robustez, ec. (dualgen)] y promediando sobre la U(1) y las traslaciones,
    p(n, Theta) = min_w [lambda_max(sum_j w_j^Theta) - n tr(w)/d^2] / n   con   tr(w tau) = 1,
con w hermítico de dos sitios e invariante por la U(1), el mismo en todas las aristas (girado). Es un problema convexo
(lambda_max es convexa en w) que se resuelve con un método de planos de corte (Kelley) con región de confianza; el
subgradiente es <psi|d H/d c_k|psi> con psi el autovector de lambda_max.

Funciones: base_u1, anillo_base, dual. Uso desde otros guiones.
"""
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
from scipy.sparse.linalg import eigsh


def base_u1(q):
    """Base real de los operadores hermíticos de dos sitios invariantes por la U(1) (índice d*a + b)."""
    d = len(q)
    carga = np.array([q[a] + q[b] for a in range(d) for b in range(d)])
    B = []
    for x in range(d * d):
        E = np.zeros((d * d, d * d), complex)
        E[x, x] = 1
        B.append(E)
        for y in range(x + 1, d * d):
            if carga[x] == carga[y]:
                E = np.zeros((d * d, d * d), complex)
                E[x, y] = E[y, x] = 1
                B.append(E)
                E = np.zeros((d * d, d * d), complex)
                E[x, y], E[y, x] = 1j, -1j
                B.append(E)
    return B


def anillo(n, d, q, w, theta):
    """sum_j w_j^Theta en el espacio completo (d^n), con el giro en el segundo sitio de cada arista."""
    N = d**n
    codes = np.arange(N, dtype=np.int64)
    pw = d ** np.arange(n, dtype=np.int64)
    dig = (codes[None, :] // pw[:, None]) % d
    phi = theta / n
    rows, cols, vals = [], [], []
    for i in range(n):
        j = (i + 1) % n
        idx = d * dig[i] + dig[j]
        for x in range(d * d):
            sel = np.nonzero(idx == x)[0]
            if sel.size == 0:
                continue
            a, b = divmod(x, d)
            for y in range(d * d):
                v = w[y, x]
                if v == 0:
                    continue
                a2, b2 = divmod(y, d)
                rows.append(codes[sel] + (a2 - a) * pw[i] + (b2 - b) * pw[j])
                cols.append(sel)
                vals.append(np.full(sel.size, v * np.exp(-1j * phi * (q[b2] - q[b]))))
    return sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N)).tocsr()


def lmax(H):
    if H.shape[0] <= 300:
        w, v = np.linalg.eigh(H.toarray())
        return w[-1], v[:, -1]
    w, v = eigsh(H, k=1, which="LA", tol=1e-13)
    return w[0], v[:, 0]


def _tope(H, kv):
    """Los kv autovalores mayores (orden decreciente) y sus autovectores."""
    from scipy.sparse.linalg import ArpackNoConvergence
    if H.shape[0] <= 800:
        w, v = np.linalg.eigh(H.toarray())
        return w[::-1][:kv], v[:, ::-1][:, :kv]
    try:
        w, v = eigsh(H, k=kv, which="LA", tol=1e-12, ncv=min(H.shape[0] - 1, max(4 * kv, 60)),
                     maxiter=20 * H.shape[0])
    except ArpackNoConvergence:  # reintento con un subespacio mayor y otro vector inicial
        rng = np.random.default_rng(0)
        w, v = eigsh(H, k=kv, which="LA", tol=1e-11, ncv=min(H.shape[0] - 1, 8 * kv),
                     maxiter=50 * H.shape[0], v0=rng.normal(size=H.shape[0]) + 0j)
    o = np.argsort(w)[::-1]
    return w[o], v[:, o]


def _suave(n, M, t, c0, Nz, B, maxit=2000, verbose=False, temps=(1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7)):
    """lambda_max suavizado, F_T = T log sum_i exp(lambda_i / T) sobre los kv autovalores mayores, con BFGS y T
    decreciente (arranque en caliente). F_T es convexa y suave, y lambda_max <= F_T <= lambda_max + T log kv, así que
    min F_T - T log kv <= p <= f(c_final): una horquilla certificada (si BFGS converge y los kv autovalores bastan).
    Devuelve (f(c_final), w, cota_inferior)."""
    from scipy.optimize import minimize
    N = M[0].shape[0]
    kv = min(N - 1, max(2 * n, 8))
    kz = Nz.shape[1]
    cache = {}

    def espectro(z):
        key = z.tobytes()
        if key not in cache:
            c = c0 + Nz @ z
            H = sum(ck * Mk for ck, Mk in zip(c, M) if ck != 0)
            w, V = _tope(H, kv)
            G = np.array([[np.real(np.vdot(V[:, i], Mk @ V[:, i])) for i in range(kv)] for Mk in M])  # m x kv
            cache.clear()
            cache[key] = (c, w, G)
        return cache[key]

    def FT(z, T):
        c, w, G = espectro(z)
        x = (w - w[0]) / (n * T)
        e = np.exp(x)
        val = w[0] / n + T * np.log(e.sum()) - t @ c
        pesos = e / e.sum()
        grad = G @ pesos / n - t
        return val, Nz.T @ grad

    z = np.zeros(kz)
    inf = -np.inf
    for T in temps:
        res = minimize(lambda zz: FT(zz, T), z, jac=True, method="BFGS", options={"gtol": 1e-13, "maxiter": maxit})
        z = res.x
        inf = max(inf, res.fun - T * np.log(kv))
        c, w, _ = espectro(z)
        sup = w[0] / n - t @ c
        if verbose:
            print(f"  T={T:.0e}: [{inf:.12f}, {sup:.12f}] nit={res.nit}", flush=True)
    c = c0 + Nz @ z
    return sup, sum(ck * Bk for ck, Bk in zip(c, B)), inf

def dual(n, d, q, tau, theta, w0=None, tol=1e-11, maxit=400, radio=None, verbose=False, metodo="kelley"):
    """p(n, Theta) por planos de corte. Devuelve (p, w_opt, iteraciones)."""
    B = base_u1(q)
    m = len(B)
    a = np.array([np.real(np.trace(Bk @ tau)) for Bk in B])  # restricción a.c = 1
    t = np.array([np.real(np.trace(Bk)) for Bk in B]) / d**2  # término lineal
    M = [anillo(n, d, q, Bk, theta) for Bk in B]
    # parametrización afín: c = c0 + N z con a.c = 1
    if w0 is None:
        c0 = a / (a @ a)
    else:
        c0 = np.array([np.real(np.trace(Bk.conj().T @ w0)) / np.real(np.trace(Bk.conj().T @ Bk)) for Bk in B])
        c0 = c0 / (a @ c0)
    _, _, Vt = np.linalg.svd(a[None, :])
    Nz = Vt[1:].T  # m x (m-1), base del núcleo de a
    def f(z):
        c = c0 + Nz @ z
        H = sum(ck * Mk for ck, Mk in zip(c, M) if ck != 0)
        lam, psi = lmax(H)
        g = np.array([np.real(np.vdot(psi, Mk @ psi)) for Mk in M])
        val = (lam - n * (t @ c)) / n
        grad = (g - n * t) / n
        return val, Nz.T @ grad
    k = m - 1
    if metodo == "suave":
        return _suave(n, M, t, c0, Nz, B, verbose=verbose)
    if metodo == "bfgs":  # BFGS sobre la función convexa no suave (Lewis-Overton): rápido en la práctica
        from scipy.optimize import minimize
        res = minimize(lambda z: f(z), np.zeros(k), jac=True, method="BFGS",
                       options={"gtol": 1e-12, "maxiter": maxit})
        c = c0 + Nz @ res.x
        return res.fun, sum(ck * Bk for ck, Bk in zip(c, B)), res.nit
    z = np.zeros(k)
    fz, gz = f(z)
    cortes = [(z.copy(), fz, gz)]
    mejor = (fz, z.copy())
    R = radio if radio is not None else 1.0
    for it in range(maxit):
        # modelo: min s  s.t.  s >= f_i + g_i.(x - z_i),  |x - z_best| <= R
        A_ub = np.array([np.r_[g, -1.0] for (_, _, g) in cortes])
        b_ub = np.array([g @ zi - fi for (zi, fi, g) in cortes])
        zb = mejor[1]
        bounds = [(zb[i] - R, zb[i] + R) for i in range(k)] + [(None, None)]
        res = linprog(np.r_[np.zeros(k), 1.0], A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if res.x is None:  # fallo numérico del LP: se reduce la región y se descartan los cortes viejos
            R *= 0.5
            cortes = cortes[-2 * k:]
            if R < 1e-9:
                break
            continue
        x = res.x[:k]
        modelo = res.x[k]
        fx, gx = f(x)
        cortes.append((x.copy(), fx, gx))
        brecha = mejor[0] - modelo
        if fx < mejor[0] - 0.1 * brecha:  # paso serio
            mejor = (fx, x.copy())
            if np.max(np.abs(x - zb)) > 0.9 * R:
                R *= 2
        else:
            R = max(R * 0.5, 1e-9)
        if verbose and it % 20 == 0:
            print(f"  it {it}: mejor={mejor[0]:.12f} brecha={brecha:.2e} R={R:.1e} cortes={len(cortes)}")
        if brecha < tol and R < 1.0:
            break
        if len(cortes) > 6 * k:  # se conservan los cortes activos y los recientes
            cortes = cortes[-4 * k:]
    c = c0 + Nz @ mejor[1]
    w = sum(ck * Bk for ck, Bk in zip(c, B))
    return mejor[0], w, it
