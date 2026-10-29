"""Ecuaciones de Bethe del anillo XXZ con flujo (sección 5.4 del artículo).

    n p(l_j) - sum_k theta(l_j - l_k) = 2 pi I_j + Phi,
    E0 = n Delta / 4 + sum_j e(l_j),

con M = floor(n/2) rapideces, para Delta > 0. Tres regímenes:

    Delta = cos(gamma) < 1:  p = 2 atan(cot(g/2) tanh l),  theta = 2 atan(cot g tanh x),
                             e = -sin^2 g / (cosh 2l - cos g)
    Delta = 1:               p = 2 atan(2 l),  theta = 2 atan(x),  e = -2/(4 l^2 + 1)
    Delta = cosh(eta) > 1:   p = 2 atan(coth(eta/2) tan l),  theta = 2 atan(coth eta tan x),
                             e = -sinh^2 eta / (cosh eta - cos 2l)

Números cuánticos: n par, I_j = j - (M+1)/2. n impar: M+1 vacantes consecutivas
simétricas; dejar libre la de un extremo u otro da las dos ramas del doblete de
la cúspide en Phi = 0.

Hamiltoniano: H = sum [1/2 (e^{i Phi/n} S+ S- + h.c.) + Delta Sz Sz], el mismo de
`holonomy.xxz`; la concordancia se comprueba en tests/test_bethe.py.
"""
import numpy as np


# Si la búsqueda lineal de Newton ya no reduce el residuo, se acepta la
# solución solo si está en el suelo de redondeo (medido: ~5e-16 n).
STALL_TOL = 1e-11


def _sech2(x):
    """sech(x)^2 sin desbordamiento."""
    t = np.exp(-2 * np.abs(x))
    return 4 * t / (1 + t) ** 2


class Regime:
    """Funciones p, theta, e y sus derivadas para un Delta dado."""

    def __init__(self, Delta, iso_tol=1e-14):
        self.Delta = Delta
        if abs(Delta - 1.0) <= iso_tol:
            self.kind = "iso"
        elif Delta < 1.0:
            if Delta <= 0.0:
                # Con flujo pi la rapidez extrema escapa a infinito en Delta = 0;
                # para Delta < 0 haría falta otra parametrización. Todos los
                # óptimos con n >= 6 tienen Delta* cerca de 1.
                raise ValueError("Bethe implementado para Delta > 0")
            self.kind = "planar"
            g = np.arccos(Delta)
            self.g = g
            self.a = 1.0 / np.tan(g / 2)
            self.b = 1.0 / np.tan(g)
            self.sg2 = np.sin(g) ** 2
            self.q = np.sin(g / 2) ** 2
        else:
            self.kind = "axial"
            eta = np.arccosh(Delta)
            self.eta = eta
            self.c = 1.0 / np.tanh(eta / 2)
            self.d = 1.0 / np.tanh(eta)
            self.sh2 = np.sinh(eta) ** 2
            self.r = np.sinh(eta / 2) ** 2

    # -- momento -------------------------------------------------------------
    def p(self, l):
        if self.kind == "iso":
            return 2 * np.arctan(2 * l)
        if self.kind == "planar":
            return 2 * np.arctan(self.a * np.tanh(l))
        return 2 * np.arctan2(self.c * np.sin(l), np.cos(l))

    def dp(self, l):
        if self.kind == "iso":
            return 4.0 / (1 + 4 * l * l)
        if self.kind == "planar":
            return 2 * self.a * _sech2(l) / (1 + self.a ** 2 * np.tanh(l) ** 2)
        return 2 * self.c / (np.cos(l) ** 2 + self.c ** 2 * np.sin(l) ** 2)

    def p_inv(self, q):
        if self.kind == "iso":
            q = np.clip(q, -np.pi * (1 - 1e-15), np.pi * (1 - 1e-15))
            return 0.5 * np.tan(q / 2)
        if self.kind == "planar":
            lim = np.pi - self.g
            q = np.clip(q, -lim * (1 - 1e-12), lim * (1 - 1e-12))
            return np.arctanh(np.tan(q / 2) / self.a)
        q = np.clip(q, -np.pi * (1 - 1e-15), np.pi * (1 - 1e-15))
        return np.arctan(np.tan(q / 2) / self.c)

    # -- desfase -------------------------------------------------------------
    def theta(self, x):
        if self.kind == "iso":
            return 2 * np.arctan(x)
        if self.kind == "planar":
            return 2 * np.arctan(self.b * np.tanh(x))
        return 2 * np.arctan2(self.d * np.sin(x), np.cos(x))

    def dtheta(self, x):
        if self.kind == "iso":
            return 2.0 / (1 + x * x)
        if self.kind == "planar":
            return 2 * self.b * _sech2(x) / (1 + self.b ** 2 * np.tanh(x) ** 2)
        return 2 * self.d / (np.cos(x) ** 2 + self.d ** 2 * np.sin(x) ** 2)

    # -- energía desnuda -----------------------------------------------------
    def e(self, l):
        if self.kind == "iso":
            return -2.0 / (4 * l * l + 1)
        if self.kind == "planar":
            # cosh 2l - cos g = 2 (tanh^2 l + sin^2(g/2) sech^2 l) / sech^2 l, sin
            # cancelaciones: cerca de Delta = 1 ambos sumandos son ~1e-6 y la
            # forma 2 - sech^2 l (1 + cos g) perdería seis cifras por rapidez.
            s2 = _sech2(l)
            return -self.sg2 * s2 / (2 * (np.tanh(l) ** 2 + self.q * s2))
        # cosh eta - cos 2l = 2 sinh^2(eta/2) + 2 sin^2 l
        return -self.sh2 / (2 * (self.r + np.sin(l) ** 2))

    def de(self, l):
        if self.kind == "iso":
            return 16 * l / (4 * l * l + 1) ** 2
        if self.kind == "planar":
            s2 = _sech2(l)
            t = np.tanh(l)
            return self.sg2 * t * s2 / (t * t + self.q * s2) ** 2
        return 2 * self.sh2 * np.sin(2 * l) / (2 * (self.r + np.sin(l) ** 2)) ** 2


def quantum_numbers(n, branch=0):
    """I_j del estado fundamental. branch = 0 (n par) o +1/-1 (n impar)."""
    M = n // 2
    if n % 2 == 0:
        return np.arange(1, M + 1) - (M + 1) / 2
    vac = np.arange(1, M + 2) - (M + 2) / 2
    if branch == +1:
        return vac[1:]
    if branch == -1:
        return vac[:-1]
    raise ValueError("n impar: branch debe ser +1 o -1")


class BetheSolver:
    """Resuelve las ecuaciones de Bethe con arranque en caliente."""

    def __init__(self, n, tol=1e-13, maxiter=100):
        self.n = n
        self.tol = tol
        self.maxiter = maxiter
        self._cache = {}

    # ------------------------------------------------------------------
    def _residual(self, reg, lam, I, Phi):
        with np.errstate(invalid="ignore", over="ignore"):
            diff = lam[:, None] - lam[None, :]
            return self.n * reg.p(lam) - reg.theta(diff).sum(axis=1) - 2 * np.pi * I - Phi

    def _jacobian(self, reg, lam):
        diff = lam[:, None] - lam[None, :]
        dt = reg.dtheta(diff)
        np.fill_diagonal(dt, 0.0)
        J = dt.copy()
        J[np.diag_indices_from(J)] = self.n * reg.dp(lam) - dt.sum(axis=1)
        return J

    def _initial(self, reg, I, Phi, iters=400, mix=0.5):
        lam = reg.p_inv((2 * np.pi * I + Phi) / self.n)
        for _ in range(iters):
            diff = lam[:, None] - lam[None, :]
            new = reg.p_inv((2 * np.pi * I + Phi + reg.theta(diff).sum(axis=1)) / self.n)
            if np.max(np.abs(new - lam)) < 1e-10 * (1 + np.max(np.abs(lam))):
                return new
            lam = (1 - mix) * lam + mix * new
        return lam

    def _newton(self, reg, lam, I, Phi):
        f = self._residual(reg, lam, I, Phi)
        norm = np.max(np.abs(f))
        for _ in range(self.maxiter):
            if not np.all(np.isfinite(f)):
                return None
            if norm < self.tol * self.n:
                return lam
            J = self._jacobian(reg, lam)
            try:
                step = np.linalg.solve(J, -f)
            except np.linalg.LinAlgError:
                return None
            t = 1.0
            while t > 1e-6:
                trial = lam + t * step
                ft = self._residual(reg, trial, I, Phi)
                nt = np.max(np.abs(ft))
                if np.isfinite(nt) and nt < norm:
                    break
                t /= 2
            else:
                return lam if norm < STALL_TOL * self.n else None
            lam, f, norm = trial, ft, nt
        return lam if norm < STALL_TOL * self.n else None

    def solve(self, Delta, Phi, branch=0):
        """Rapideces del estado de números cuánticos I(branch).

        Arranque: la última solución de la misma rama, traducida a través de
        los momentos p_j (continuos en Delta, también a través de Delta = 1).
        Si Newton no converge desde ahí, continuación en (Delta, Phi) desde la
        última solución en pasos cada vez más finos; como último recurso,
        iteración de punto fijo amortiguada desde cero.
        """
        reg = Regime(Delta)
        I = quantum_numbers(self.n, branch)
        lam = None
        if branch in self._cache:
            old_reg, old_lam, old_Phi = self._cache[branch]
            lam = self._newton(reg, self._translate(old_reg, old_lam, reg), I, Phi)
            for steps in (4, 16):
                if lam is not None:
                    break
                lam = self._continuation(old_reg, old_lam, old_Phi, Delta, Phi, I, steps)
        if lam is None:
            lam = self._newton(reg, self._initial(reg, I, Phi), I, Phi)
        if lam is None:
            raise RuntimeError(f"Bethe no converge: n={self.n} Delta={Delta} Phi={Phi}")
        self._cache[branch] = (reg, lam, Phi)
        return reg, lam

    @staticmethod
    def _translate(old_reg, old_lam, reg):
        """Arranque para `reg` desde una solución de `old_reg`: se conservan
        los momentos p_j, que dependen suavemente de Delta (las rapideces no:
        cerca de Delta = 1 escalan como gamma o eta)."""
        if old_reg.Delta == reg.Delta:
            return old_lam.copy()
        return reg.p_inv(old_reg.p(old_lam))

    def _continuation(self, old_reg, lam, old_Phi, Delta, Phi, I, steps):
        reg = old_reg
        for t in np.linspace(0, 1, steps + 1)[1:]:
            new_reg = Regime(old_reg.Delta + t * (Delta - old_reg.Delta))
            lam = self._newton(new_reg, self._translate(reg, lam, new_reg), I,
                               old_Phi + t * (Phi - old_Phi))
            if lam is None:
                return None
            reg = new_reg
        return lam

    # ------------------------------------------------------------------
    def energy(self, Delta, Phi, branch=0):
        reg, lam = self.solve(Delta, Phi, branch)
        return self.n * Delta / 4 + reg.e(lam).sum()

    def energy_and_flux_derivative(self, Delta, Phi, branch=0):
        """(E0, dE0/dPhi) por diferenciación implícita: J dl/dPhi = 1."""
        reg, lam = self.solve(Delta, Phi, branch)
        J = self._jacobian(reg, lam)
        dlam = np.linalg.solve(J, np.ones_like(lam))
        return self.n * Delta / 4 + reg.e(lam).sum(), float(reg.de(lam) @ dlam)

    # ------------------------------------------------------------------
    def ground_branch(self):
        """Rama del estado fundamental para Phi en [0, pi].

        n par: la única. n impar: la -1 (vacante libre en el extremo superior);
        en Phi = 0 es degenerada con la +1, y para Phi > 0 es la de menor
        energía (comprobado contra diagonalización exacta en tests/).
        """
        return 0 if self.n % 2 == 0 else -1

    def eps(self, Delta, Phi):
        """eps_n = -E0/n para Phi en [0, pi].

        Cerca de Delta = -1 y para n pequeño el estado fundamental puede ser el
        ferromagnético (energía n Delta/4), que no describe esta rama.
        """
        E = self.energy(Delta, Phi, self.ground_branch())
        return -min(E, self.n * Delta / 4) / self.n

    def cusp_slope(self, Delta):
        """s_n = d eps / d|Phi - Phi_c| en la cúspide, por derivada implícita.

        n par: cúspide en Phi = pi; s_n = (1/n) dE/dPhi en pi por la izquierda.
        n impar: cúspide en Phi = 0; s_n = -(1/n) dE/dPhi en 0 por la derecha.
        """
        n = self.n
        b = self.ground_branch()
        if n % 2 == 0:
            return self.energy_and_flux_derivative(Delta, np.pi, b)[1] / n
        return -self.energy_and_flux_derivative(Delta, 0.0, b)[1] / n


_solvers = {}


def solver(n):
    """BetheSolver compartido por n (conserva el arranque en caliente)."""
    if n not in _solvers:
        _solvers[n] = BetheSolver(n)
    return _solvers[n]


def eps(n, Delta, Phi):
    return solver(n).eps(Delta, Phi)


def cusp_slope(n, Delta):
    return solver(n).cusp_slope(Delta)
