"""Ecuaciones de Bethe del anillo XXZ con flujo en variables de momento, -1 < Delta < 1.

Mismas ecuaciones y números cuánticos que `holonomy.bethe`,

    n p_j - sum_k theta(p_j, p_k) = 2 pi I_j + Phi,     E0 = n Delta / 4 + sum_j (-cos p_j - Delta),

pero con los momentos p_j como incógnitas en lugar de las rapideces. Con
Delta = cos(gamma) y u = tanh(lambda) = tan(p/2) tan(gamma/2), p recorre la recta
real de rapideces (|p| < pi - gamma) y la recta Im lambda = pi/2 (|p| > pi - gamma)
sin pasar por infinito; la rapidez que escapa a infinito con flujo pi en Delta = 0 es
aquí p = pi - gamma, un punto regular. Así llega a Delta <= 0, fuera del alcance de
`holonomy.bethe`. El desfase es

    theta(p1, p2) = 2 atan(cot(gamma) N / D),   N = t sin((p1 - p2)/2),
    D = cos(p1/2) cos(p2/2) - t^2 sin(p1/2) sin(p2/2),   t = tan(gamma/2),

que es 2 atan(cot(gamma) tanh(lambda1 - lambda2)). La fórmula solo fija theta módulo
2 pi (la rama salta al cruzar D = 0, cuando un momento está en la recta Im lambda = pi/2),
y los números cuánticos solo tienen sentido con theta continuo. Por eso se sigue la rama:
cada theta_jk se toma como el representante más cercano a su valor en el punto anterior
de la continuación (theta_ref), y el residuo no se reduce módulo 2 pi. Reducirlo deja que
Newton cambie los I_j sin que se note y caiga en soluciones espurias con momentos repetidos.

Arranque: Delta = 0 (fermiones libres, theta = 0) da p_j = (2 pi I_j + Phi)/n
exactamente; se continúa en Delta desde ahí o desde la solución guardada más cercana con
el mismo (n, rama, Phi), con un predictor tangente (J dp/dDelta = -dF/dDelta).

Pares complejos. Con Delta < 0 y flujo, por debajo de Delta_c(Phi) las dos raíces extremas
chocan en p = pi - gamma (lambda = infinito) y siguen como un par complejo conjugado: con n par,
pi - gamma_c = Phi/4 (Delta_c = -1/sqrt 2 con Phi = pi); con n impar y la rama -1,
pi - gamma_c = (pi - Phi)/5. Más abajo se forman más pares (el segundo en Phi/8 o (pi - Phi)/9). Las ecuaciones siguen siendo las mismas, con p complejos (theta
por -i log, misma rama por continuidad). Cuando la continuación real se atasca, `_cruce` abre
las dos reales más próximas en m +- i h y da el paso con Newton complejo (o cierra un par al
volver); los pares ya formados se conservan. Se comprueba frente a la diagonalización exacta (tests). La energía
-cos p - Delta coincide con la de `holonomy.bethe` (e(lambda) = -sin^2 gamma /
(cosh 2 lambda - cos gamma)).
"""
import numpy as np

from .bethe import quantum_numbers


class MomentumSolver:
    """Resuelve las ecuaciones de Bethe en momentos para -1 < Delta < 1, con un par complejo
    conjugado cuando hace falta (ver `_cruce`)."""

    def __init__(self, n, tol=1e-12, maxiter=40, dstep=0.05):
        self.n = n
        self.tol = tol
        self.maxiter = maxiter
        self.dstep = dstep
        self._cache = {}  # (branch, Phi) -> lista de (Delta, p, theta)

    @staticmethod
    def _consts(Delta):
        if not -1 < Delta < 1:
            raise ValueError("bethe_p: -1 < Delta < 1")
        g = np.arccos(Delta)
        return g, 1 / np.tan(g), np.tan(g / 2)

    @staticmethod
    def _parts(p, t):
        a, b = p[:, None], p[None, :]
        sa, ca, sb, cb = np.sin(a / 2), np.cos(a / 2), np.sin(b / 2), np.cos(b / 2)
        N = t * np.sin((a - b) / 2)
        D = ca * cb - t * t * sa * sb
        return sa, ca, sb, cb, N, D

    def _theta(self, p, cg, t, ref):
        _, _, _, _, N, D = self._parts(p, t)
        with np.errstate(divide="ignore", invalid="ignore"):
            if np.iscomplexobj(p):
                # 2 atan(cg N / D) = -i log((D + i cg N) / (D - i cg N)), también con p complejos
                raw = -1j * np.log((D + 1j * cg * N) / (D - 1j * cg * N))
            else:
                raw = 2 * np.arctan2(cg * N, D)
        th = raw + 2 * np.pi * np.round(np.real(ref - raw) / (2 * np.pi))
        np.fill_diagonal(th, 0.0)
        return th

    def _residual(self, p, I, Phi, th):
        return self.n * p - th.sum(axis=1) - 2 * np.pi * I - Phi

    def _derivs(self, p, cg, t):
        """(d theta/d p1, d theta/d p2) en matrices, sin diagonal."""
        sa, ca, sb, cb, N, D = self._parts(p, t)
        a, b = p[:, None], p[None, :]
        dN1 = 0.5 * t * np.cos((a - b) / 2)
        dD1 = -0.5 * sa * cb - 0.5 * t * t * ca * sb
        dD2 = -0.5 * ca * sb - 0.5 * t * t * sa * cb
        den = D * D + cg * cg * N * N
        np.fill_diagonal(den, 1.0)
        d1 = 2 * cg * (D * dN1 - N * dD1) / den
        d2 = 2 * cg * (-D * dN1 - N * dD2) / den
        np.fill_diagonal(d1, 0.0)
        np.fill_diagonal(d2, 0.0)
        return d1, d2, N, D, den

    def _jacobian(self, p, cg, t):
        d1, d2, *_ = self._derivs(p, cg, t)
        J = -d2
        J[np.diag_indices_from(J)] = self.n - d1.sum(axis=1)
        return J

    def _dF_dDelta(self, p, g, cg, t):
        """Derivada del residuo respecto a Delta a p fijos (theta depende de cg y t)."""
        sa, ca, sb, cb, N, D = self._parts(p, t)
        dg = -1 / np.sin(g)                      # d gamma / d Delta
        dcg = -dg / np.sin(g) ** 2               # d cot(gamma) / d Delta
        dt = dg * 0.5 / np.cos(g / 2) ** 2       # d tan(gamma/2) / d Delta
        dN = N / t * dt
        dD = -2 * t * dt * sa * sb
        den = D * D + cg * cg * N * N
        np.fill_diagonal(den, 1.0)
        dth = 2 * (dcg * N * D + cg * (dN * D - N * dD)) / den
        np.fill_diagonal(dth, 0.0)
        return -dth.sum(axis=1)

    def _newton(self, p, I, Phi, Delta, ref):
        g, cg, t = self._consts(Delta)
        th = self._theta(p, cg, t, ref)
        f = self._residual(p, I, Phi, th)
        norm = np.max(np.abs(f))
        for _ in range(self.maxiter):
            if norm < self.tol * self.n:
                return p, th
            try:
                step = np.linalg.solve(self._jacobian(p, cg, t), -f)
            except np.linalg.LinAlgError:
                return None
            s = 1.0
            while s > 1e-4:
                trial = p + s * step
                tht = self._theta(trial, cg, t, th)
                ft = self._residual(trial, I, Phi, tht)
                nt = np.max(np.abs(ft))
                if nt < norm:
                    break
                s /= 2
            else:
                return (p, th) if norm < 1e-10 * self.n else None
            p, th, f, norm = trial, tht, ft, nt
        return (p, th) if norm < 1e-10 * self.n else None

    def _ok(self, p):
        """Rechaza soluciones con momentos repetidos (no son autoestados) o sin simetría conjugada."""
        if len(p) < 2:
            return True
        d = np.abs(p[:, None] - p[None, :])
        np.fill_diagonal(d, np.inf)
        if np.min(d) < 1e-6 / self.n:
            return False
        if np.iscomplexobj(p):
            # el conjunto de momentos debe ser cerrado por conjugación (energía real)
            c = np.abs(np.conj(p)[:, None] - p[None, :]).min(axis=1)
            return np.max(c) < 1e-8
        return True

    @staticmethod
    def _limpia(p):
        """Pone a cero las partes imaginarias de redondeo; vuelve a real si no queda ningún par."""
        if not np.iscomplexobj(p):
            return p
        p = np.where(np.abs(p.imag) < 1e-12, p.real + 0j, p)
        return p.real.copy() if np.all(p.imag == 0) else p

    @staticmethod
    def _pares(q):
        """Índices (a, b) de los pares complejos conjugados de q, con Im q_a > 0."""
        if not np.iscomplexobj(q):
            return []
        arriba = np.where(q.imag > 0)[0]
        abajo = list(np.where(q.imag < 0)[0])
        pares = []
        for a in arriba:
            b = min(abajo, key=lambda k: abs(q[k] - np.conj(q[a])))
            abajo.remove(b)
            pares.append((a, b))
        return pares

    def _cruce(self, q, tq, I, Phi, Dc, Delta):
        """Paso a través de un umbral: dos raíces reales se juntan en p = pi - gamma y forman un par
        complejo conjugado, o un par se cierra en dos reales. Puede haber ya otros pares, que se
        conservan. Al bajar en Delta se prueba antes abrir un par nuevo (con las dos reales más
        próximas); al subir, cerrar el par más estrecho. Devuelve (Delta, p, theta) o None."""
        sgn = np.sign(Delta - Dc)
        qc = q.astype(complex)
        opciones = []
        reales = np.where(qc.imag == 0)[0]
        if len(reales) >= 2:
            order = reales[np.argsort(qc[reales].real)]
            k = int(np.argmin(np.diff(qc[order].real)))
            i1, i2 = order[k], order[k + 1]
            opciones.append(("abrir", i1, i2, (qc[i1].real + qc[i2].real) / 2,
                             abs(qc[i2].real - qc[i1].real) / 2))
        for a, b in sorted(self._pares(qc), key=lambda ab: qc[ab[0]].imag):
            opciones.append(("cerrar", a, b, qc[a].real, qc[a].imag))
        if sgn > 0:
            opciones.sort(key=lambda o: o[0] != "cerrar")
        ref = tq.astype(complex)
        # Cerca del choque el par se abre como una raíz cuadrada (separación ~ |Delta - Delta_c|^1/2,
        # real a un lado e imaginaria al otro) y, con n grande, es estrecho: se prueban pasos desde
        # 1e-6 y aperturas alrededor de la separación del último punto.
        for tipo, i1, i2, m, h in opciones:
            for step in (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2):
                Dn = Dc + sgn * min(step, abs(Delta - Dc))
                for f in (1.0, 2.0, 0.5, 4.0, 0.25):
                    trial = qc.copy()
                    if tipo == "abrir":
                        trial[i1], trial[i2] = m - 1j * f * h, m + 1j * f * h
                    else:
                        trial[i1], trial[i2] = m - f * h, m + f * h
                    r = self._newton(trial, I, Phi, Dn, ref)
                    if r is not None and self._ok(r[0]):
                        return Dn, self._limpia(r[0]), r[1]
                if step >= abs(Delta - Dc):
                    break
        return None

    def _salto(self, q, tq, I, Phi, Dc, Delta):
        """Salta un punto casi singular que no es un umbral de par (por ejemplo, dos pares que se
        acercan entre sí): Newton desde el predictor con pasos de 1e-3 a 1e-2."""
        g, cg, t = self._consts(Dc)
        try:
            dp = np.linalg.solve(self._jacobian(q, cg, t), -self._dF_dDelta(q, g, cg, t))
        except np.linalg.LinAlgError:
            dp = np.zeros_like(q)
        # Cerca del punto el jacobiano está casi degenerado y el predictor tangente puede ser malo:
        # se prueba también sin predictor.
        for step in (1e-3, 2.5e-3, 5e-3, 1e-2):
            Dn = Dc + np.sign(Delta - Dc) * min(step, abs(Delta - Dc))
            for arranque in (q + (Dn - Dc) * dp, q):
                r = self._newton(arranque, I, Phi, Dn, tq)
                if r is not None and self._ok(r[0]) and np.max(np.abs(r[1] - tq)) <= 2.5:
                    return Dn, self._limpia(r[0]), r[1]
            if step >= abs(Delta - Dc):
                break
        return None

    def _continua(self, Dc, q, tq, Delta, I, Phi, hmax):
        h = hmax
        while abs(Delta - Dc) > 1e-15:
            Dn = Dc + np.sign(Delta - Dc) * min(h, abs(Delta - Dc))
            # predictor tangente desde (Dc, q)
            g, cg, t = self._consts(Dc)
            try:
                dp = np.linalg.solve(self._jacobian(q, cg, t), -self._dF_dDelta(q, g, cg, t))
            except np.linalg.LinAlgError:
                dp = np.zeros_like(q)
            r = self._newton(q + (Dn - Dc) * dp, I, Phi, Dn, tq)
            if r is None or not self._ok(r[0]) or np.max(np.abs(r[1] - tq)) > 2.5:
                h /= 4
                if h < 1e-5:
                    c = self._cruce(q, tq, I, Phi, Dc, Delta) or self._salto(q, tq, I, Phi, Dc, Delta)
                    if c is None:
                        return None
                    Dc, q, tq = c
                    h = 1e-3
                continue
            Dc, q, tq = Dn, self._limpia(r[0]), r[1]
            if not np.iscomplexobj(q):
                tq = np.real(tq)
            h = min(hmax, 2 * h)
        return q, tq

    def solve(self, Delta, Phi, branch=0):
        """Momentos p_j del estado de números cuánticos I(branch), por continuación en Delta.

        Si la continuación con paso máximo dstep se pierde (por ejemplo, al cruzar un umbral de par
        con un salto grande en Delta), se repite desde el mismo punto con paso máximo 0,0025."""
        I = quantum_numbers(self.n, branch)
        key = (branch, float(Phi))
        p0 = (2 * np.pi * I + Phi) / self.n
        hist = self._cache.setdefault(key, [(0.0, p0, np.zeros((len(I), len(I))))])
        D0, p, th = min(hist, key=lambda h: abs(h[0] - Delta))
        for hmax in (self.dstep, 0.0025):
            r = self._continua(D0, p.copy(), th.copy(), Delta, I, Phi, hmax)
            if r is not None:
                break
        else:
            raise RuntimeError(f"bethe_p no converge: n={self.n} Delta={Delta} Phi={Phi}")
        q, tq = r
        hist.append((float(Delta), q, tq))
        if len(hist) > 40:
            hist.pop(1)
        return q

    def energy(self, Delta, Phi, branch=0):
        p = self.solve(Delta, Phi, branch)
        return self.n * Delta / 4 + float(np.real(np.sum(-np.cos(p) - Delta)))


_solvers = {}


def solver(n):
    if n not in _solvers:
        _solvers[n] = MomentumSolver(n)
    return _solvers[n]


def eps(n, Delta, Phi):
    """eps_n = -E0/n para Phi en [0, pi] (rama del fundamental de `holonomy.bethe`).

    Como en `holonomy.bethe.eps`, incluye el estado ferromagnético (energía n Delta/4), que
    cerca de Delta = -1 y con n pequeño puede ser el fundamental.
    """
    S = solver(n)
    b = 0 if n % 2 == 0 else -1
    return -min(S.energy(Delta, Phi, b), n * Delta / 4) / n
