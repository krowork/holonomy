"""Números del artículo de rigidez que salen de los CSV de data/ (solo lectura).

Artículo (~/fisica/holonomia_rigidez.tex, «Holonomía y pegado de perspectivas
de Werner: la rigidez del testigo óptimo frente a un flujo»): todas las
secciones. No resuelve ningún SDP ni diagonaliza nada: lee los CSV y comprueba
cada cifra tal como está escrita en el artículo, con su redondeo.

Dos comprobaciones:
  1. Cada cifra contra su CSV. Modos (como en diamond_numeros.py):
       "="  el valor del CSV, redondeado a las cifras del texto, da el texto;
       "<=" cota superior escrita redondeando hacia arriba: t - u < x <= t,
            con u una unidad de la última cifra del texto;
       ">=" cota inferior escrita redondeando hacia abajo: t <= x < t + u.
     Además, cada cifra comprobada tiene que aparecer en el .tex.
  2. Ningún número sin guion: se extraen del .tex (entre \\begin{document} y
     la bibliografía, sin comentarios) todos los números decimales (con coma,
     también «0{,}25») y todos los de la forma m\\times10^{e}; cada uno tiene
     que estar entre las cifras comprobadas en 1, o entre los tiempos y picos
     de memoria de la tabla de guiones, que se comprueban contra el README
     (son medidas de las corridas, no datos de un CSV). Los enteros (tamaños,
     recuentos) y las potencias sueltas 10^{-k} (tolerancias) no se extraen;
     los recuentos que importan se comprueban aparte.
Si el .tex no está (no está en git), solo se hace la comprobación 1 contra los
CSV y se avisa.

Datos:
  §4 anillos: data/rigidez_anillos.csv, data/edges_ed.csv (anchura de la
     ventana de los qubits y v en 0 y pi, tabla 1 de anillos) y
     data/table62.csv (tabla 2 de anillos), referencias independientes de v0
     y v1;
  §5 escaleras: data/rigidez_escaleras.csv (también la columna multiplete:
     nivel superior único con f = 0, degenerado en momento con f = pi) y
     data/ladder_limit_L<L>.csv,
     ladder_limit_F0_L<L>.csv (las mismas definiciones de variación, cota,
     exponentes locales, ajustes y Shanks que rigidez_escaleras.py, importadas
     de allí);
  §6 diamantes: data/rigidez_diamantes.csv, data/diamond_ring_L<k>_igual.csv;
  §7 qudits: data/rigidez_qudit_limite.csv, rigidez_qudit_flujo.csv,
     rigidez_qudit_asint.csv;
  §9 tabla de guiones: README.md.

Uso: python scripts/rigidez_numeros.py [ruta del .tex]
     (por defecto ~/fisica-old/holonomia_rigidez.tex; 0,5 s y 0,12 GB de pico, medidos con
     /usr/bin/time -f "%M KB %e s"; sin SDP)
"""
import _common  # noqa: F401  (fija OMP_NUM_THREADS antes de numpy)
from _common import Checker

import csv
import math
import re
import sys
from pathlib import Path

import numpy as np

from diamond_numeros import cifra
from rigidez_escaleras import fits, local_exponents, shanks, witness_params
from holonomy.report import DATA

pi = math.pi
TEX = Path.home() / "fisica-old" / "holonomia_rigidez.tex"
README = Path(__file__).resolve().parent.parent / "README.md"

# Tiempos y picos de la tabla de guiones (§9): (texto en el artículo, texto en
# el README). Las cifras decimales de aquí no salen de un CSV.
MEDIDAS = [
    ("299 s; 0,51 GB", "299 s, pico 0,51 GB"),
    ("27 min; 4,3 GB", "~27 min en total"),
    ("27 min; 4,3 GB", "pico 4,3 GB"),
    ("31 s; 0,79 GB", "31 s, pico 0,79 GB"),
    ("68 s; 1,23 GB", "68 s, pico 1,23 GB"),
    ("1774 s; 1,75 GB", "1774 s en frío sin corrientes ni certificación, pico 1,75 GB"),
    ("110 s; 1,20 GB", "110 s, pico 1,20 GB"),
    ("tarda 31 s", "con el CSV escrito, 31 s"),
    ("884 s + 1101 s con pico de 6,0 GB", "costó 884 s + 1101 s (pico 6,0 GB)"),
    ("(6,6 GB", "pico 6,6 GB"),
]


def leer(name):
    with open(DATA / name, newline="") as fh:
        return list(csv.DictReader(fh))


def f(r, c):
    return float(r[c])


# ---------------------------------------------------------------------------
# Texto
# ---------------------------------------------------------------------------

def cuerpo(tex):
    """Texto entre \\begin{document} y la bibliografía, sin comentarios y
    con «{,}» como «,»."""
    s = tex.split("\\begin{document}", 1)[1].split("\\begin{thebibliography}", 1)[0]
    s = "\n".join(re.sub(r"(?<!\\)%.*", "", ln) for ln in s.splitlines())
    return s.replace("{,}", ",")


SCI = re.compile(r"(\d+(?:,\d+)?)\s*\\times\s*10\^\{?(-?\d+)\}?")
DEC = re.compile(r"(?<!\d)(\d+,\d+)(?!\d)")


def numeros(s):
    """Números del texto, sin signo: «1,7e-9» para 1{,}7\\times10^{-9} y
    «0,25» para los decimales."""
    out = [f"{m.group(1)}e{int(m.group(2))}" for m in SCI.finditer(s)]
    s = SCI.sub(" ", s)
    out += DEC.findall(s)
    return out


def sin_signo(txt):
    return txt.lstrip("-")


# ---------------------------------------------------------------------------
# Comparación con el redondeo del texto
# ---------------------------------------------------------------------------

def unidad(txt):
    """Valor y unidad de la última cifra de txt («1,7e-9» -> 1,7e-9 y 1e-10)."""
    t = txt.replace(",", ".")
    if "e" in t:
        m, e = t.split("e")
        d = len(m.split(".")[1]) if "." in m else 0
        return float(t), 10.0 ** (int(e) - d)
    d = len(t.split(".")[1]) if "." in t else 0
    return float(t), 10.0 ** (-d)


def compara(x, txt, modo):
    if modo == "=":
        return cifra(x, txt)
    t, u = unidad(txt)
    eps = 1e-12 * max(abs(t), u)
    if modo == "<=":
        return t - u < x <= t + eps
    if modo == ">=":
        return t - eps <= x < t + u
    raise ValueError(modo)


class Cifras:
    """Registro de las cifras comprobadas, con el texto del artículo."""

    def __init__(self, ck, s):
        self.ck, self.s = ck, s
        self.en_tex = set(numeros(s)) if s is not None else None
        self.vistas = set()

    def __call__(self, txt, x, msg, modo="="):
        key = sin_signo(txt)
        self.vistas.add(key)
        ok = compara(x, txt, modo)
        nota = ""
        if self.en_tex is not None and key not in self.en_tex:
            ok, nota = False, "  [no aparece en el .tex]"
        rel = {"=": "->", "<=": "<=", ">=": ">="}[modo]
        self.ck.check(ok, f"{msg}: {x:.6e} {rel} «{txt}»{nota}")


# ---------------------------------------------------------------------------
# §1 y §4: anillos
# ---------------------------------------------------------------------------

def anillos(ck, num):
    rows = leer("rigidez_anillos.csv")
    pi_rows = {int(r["n"]): r for r in rows if abs(f(r, "giro") - pi) < 1e-12}
    p2_rows = {int(r["n"]): r for r in rows if abs(f(r, "giro") - pi / 2) < 1e-12}
    ck.check(sorted(pi_rows) == list(range(6, 21)) and sorted(p2_rows) == list(range(6, 21)),
             f"§4: giros pi y pi/2 en n = 6..20 ({len(rows)} filas)")
    dw = [f(r, "D_W") for r in pi_rows.values()]
    num("0,2804", min(dw), "§1, §4: D_W mínima (n = 20)")
    num("0,3066", max(dw), "§1, §4: D_W máxima (n = 6)")
    dphi = {n: f(r, "curv_envolvente") / f(r, "b") for n, r in pi_rows.items()}
    ck.check(round(min(dphi.values())) == 18 and round(max(dphi.values())) == 65,
             f"§1, §4: dPhi*/dTheta de {min(dphi.values()):.2f} a {max(dphi.values()):.2f} («18 a 65»)")
    r20 = pi_rows[20]
    marg = max(1 - f(r20, "cociente_inf"), f(r20, "cociente_sup") - 1)
    ck.check(marg < 0.02, f"§1, §4: giro pi, n = 20: Kohn e inversa a {100 * marg:.2f} % de la caída (< 2 %)")
    tabla = {6: ("0,3066", "1,0000", "-0,4088", "-26,554", "64,95"),
             7: ("0,2852", "1,5372", "-0,3225", "-20,594", "63,86"),
             10: ("0,2906", "1,0000", "-0,3875", "-11,219", "28,95"),
             15: ("0,2832", "1,0897", "-0,3667", "-9,619", "26,23"),
             20: ("0,2804", "1,0000", "-0,3739", "-6,772", "18,11")}
    for n, (tD, tDe, tb, tc, tp) in tabla.items():
        r = pi_rows[n]
        num(tD, f(r, "D_W"), f"§4, tabla: D_W, n = {n}")
        num(tDe, f(r, "Delta0"), f"§4, tabla: Delta0, n = {n}")
        num(tb, f(r, "b"), f"§4, tabla: b = -4 D_W/den0, n = {n}")
        num(tc, f(r, "curv_envolvente"), f"§4, tabla: n^2 v'' (envolvente), n = {n}")
        num(tp, dphi[n], f"§4, tabla: dPhi*/dTheta, n = {n}")
    ab = {n: (f(r, "a") + f(r, "b")) / f(r, "a") for n, r in pi_rows.items()}
    num("0,0154", ab[6], "§4: (a+b)/a, n = 6")
    num("0,0552", ab[20], "§4: (a+b)/a, n = 20")
    ed = {int(r["n"]): f(r, "psi_over_pi") for r in leer("edges_ed.csv")}
    num("0,018587", ed[6], "§4, §7.3: ventana de los qubits (anillos, tabla 1), n = 6")
    num("0,068431", ed[20], "§4: ventana de los qubits (anillos, tabla 1), n = 20")
    num("0,026632", ed[9], "§7.3: ventana de los qubits (anillos, tabla 1), n = 9")
    num("0,877", f(pi_rows[6], "cociente_inf"), "§4: giro pi, n = 6: Kohn/caída")
    num("1,119", f(pi_rows[6], "cociente_sup"), "§4: giro pi, n = 6: inversa/caída")
    num("0,981", f(r20, "cociente_inf"), "§4: giro pi, n = 20: Kohn/caída")
    num("1,019", f(r20, "cociente_sup"), "§4: giro pi, n = 20: inversa/caída")
    ci2 = [f(r, "cociente_inf") for r in p2_rows.values()]
    num("0,302", min(ci2), "§4: giro pi/2: Kohn/caída, la menor (n = 6..20)")
    num("0,338", max(ci2), "§4: giro pi/2: Kohn/caída, la mayor (n = 6..20)")
    num("1,035", f(p2_rows[20], "cociente_sup"), "§4: giro pi/2, n = 20: inversa/caída")
    num("8,00", f(pi_rows[6], "n2_cota_lema5"), "§4: 4n/den0 (Lema 5(i) de anillos), n = 6")
    num("26,67", f(r20, "n2_cota_lema5"), "§4: 4n/den0 (Lema 5(i) de anillos), n = 20")
    num("1,8543", f(r20, "n2_caida"), "§4: caída n^2[v(0) - v(pi)], n = 20")
    num("2,2015", f(pi_rows[6], "n2_caida"), "§4: caída n^2[v(0) - v(pi)], n = 6")
    # Isotropía por paridad: Delta0 = 1 con n par, != 1 con n impar.
    dpar = max(abs(f(r, "Delta0") - 1) for n, r in pi_rows.items() if n % 2 == 0)
    dimp = min(abs(f(r, "Delta0") - 1) for n, r in pi_rows.items() if n % 2 == 1)
    ck.check(dpar < 1e-6 and dimp > 0.05,
             f"§4: Delta0 = 1 con n par (a {dpar:.1e}) y anisótropo con n impar (|Delta0 - 1| >= {dimp:.3f})")
    # v0 y v1 frente a las tablas 1 (edges_ed: v0, vpi, n <= 20) y 2 (table62:
    # Theta = 0, pi/2, pi, n <= 16) de anillos: referencia independiente de la
    # minimización en la rejilla de 7 flujos («a 10^{-14}» en §4).
    t62 = {int(r["n"]): r for r in leer("table62.csv")}
    eed = {int(r["n"]): r for r in leer("edges_ed.csv")}
    c62 = {0: "v_0", 1: "v_pi/2", 2: "v_pi"}
    ced = {0: "v0", 2: "vpi"}
    d62 = ded = 0.0
    n62 = ned = 0
    for r in rows:
        n = int(r["n"])
        for T, v in ((f(r, "Theta0"), f(r, "v0")), (f(r, "Theta1"), f(r, "v1"))):
            q = round(2 * T / pi)
            if n in t62:
                d62, n62 = max(d62, abs(f(t62[n], c62[q]) - v)), n62 + 1
            if q in ced and n in eed:
                ded, ned = max(ded, abs(f(eed[n], ced[q]) - v)), ned + 1
    ck.check(d62 <= 1e-14 and ded <= 1e-14 and n62 == 44 and ned == 45,
             f"§4: v0 y v1 frente a la tabla 2 de anillos (table62, {n62} valores, n <= 16) a {d62:.1e} y "
             f"a la tabla 1 (edges_ed, {ned} valores, n <= 20) a {ded:.1e} (<= 1e-14)")




# ---------------------------------------------------------------------------
# §5: escaleras
# ---------------------------------------------------------------------------

PSIS = [0, pi / 4, pi / 2, 3 * pi / 4, pi]
PSI_NAMES = ["0", "pi4", "pi2", "3pi4", "pi"]


def ref(fname, L):
    name = f"ladder_limit_L{L}.csv" if fname == "pi" else f"ladder_limit_F0_L{L}.csv"
    return {round(f(r, "psi"), 9): f(r, "v_max") for r in leer(name)}


def escaleras(ck, num):
    rows = leer("rigidez_escaleras.csv")
    ck.check(len(rows) == 31, f"§5: {len(rows)} testigos (31)")
    by = {(r["f"], int(r["L"]), round(f(r, "psi0"), 9)): r for r in rows}
    hol, dref = np.inf, 0.0
    for (fn, L, p0), r in by.items():
        rf = ref(fn, L)
        dref = max(dref, abs(f(r, "v_max") - rf[p0]))
        hol = min(hol, min(f(r, f"cota_{nm}") - rf[round(p, 9)] for p, nm in zip(PSIS, PSI_NAMES)))
    num("-1,3e-9", hol, "§5: holgura mínima de la cota en 31 testigos x 5 psi")
    num("1,7e-9", dref, "§5: máx |v del testigo - v de ladder_limit|")
    num("7,2e-15", max(abs(f(r, "dif_aleatorio")) for r in rows), "§5: Lanczos aleatorio, máx |dif|")
    num("9,1e-14", max(f(r, "dif_norma") for r in rows), "§5: normalización del giro, máx |N - 1|")

    # Nivel superior del testigo (columna multiplete, etiquetas k:q x mult con
    # k espines arriba y momento 2 pi q / L): único en f = 0; degenerado en
    # momento en f = pi (0 y pi con L par, salvo L = 4, psi0 = pi; dos
    # dobletes +-q con L impar), siempre en S^z = 0 (k = L).
    def mult(r):
        L = int(r["L"])
        out = []
        for tok in r["multiplete"].split():
            kq, m = tok.split("x")
            k, q = map(int, kq.split(":"))
            out.append((k - L, q % L, int(m)))
        return sorted(out)

    f0 = [r for r in rows if r["f"] == "0"]
    ck.check(len(f0) == 14 and all(mult(r) == [(0, 0, 1)] for r in f0),
             f"§5: f = 0: nivel superior único (S^z = 0, momento 0) en los {len(f0)} testigos (14)")
    fpp = [r for r in rows if r["f"] == "pi" and int(r["L"]) % 2 == 0]
    dob = [r for r in fpp if mult(r) == [(0, 0, 1), (0, int(r["L"]) // 2, 1)]]
    exc = [r for r in fpp if r not in dob]
    ck.check(len(fpp) == 8 and len(dob) == 7 and len(exc) == 1 and int(exc[0]["L"]) == 4
             and abs(f(exc[0], "psi0") - pi) < 1e-9 and mult(exc[0]) == [(0, 2, 1)],
             f"§5: f = pi, L par: nivel superior en momentos 0 y pi en {len(dob)} de {len(fpp)} testigos; "
             "el de L = 4, psi0 = pi solo en pi (7 de 8)")
    fpi = [r for r in rows if r["f"] == "pi" and int(r["L"]) % 2 == 1]

    def dos_dobletes(r):
        L, mm = int(r["L"]), mult(r)
        qs = sorted(q for _, q, _ in mm)
        return (len(mm) == 4 and all(sz == 0 and m == 1 for sz, _, m in mm)
                and all((L - q) % L in qs for q in qs) and 0 not in qs)
    esperado = {5: [1, 2, 3, 4], 7: [1, 3, 4, 6], 9: [1, 3, 6, 8]}
    ck.check(len(fpi) == 9 and all(dos_dobletes(r) for r in fpi)
             and all(sorted(q for _, q, _ in mult(r)) == esperado[int(r["L"])] for r in fpi),
             f"§5: f = pi, L impar: nivel superior en cuatro momentos, dos dobletes +-q "
             f"(+-1, +-2 de 5; +-1, +-3 de 7 y de 9), en los {len(fpi)} testigos (9)")

    ser = {}
    for fn in ("pi", "0"):
        for L in range(4, 11):
            rf = ref(fn, L)
            vs = np.array([rf[round(p, 9)] for p in PSIS])
            jmin, jmax = int(np.argmin(vs)), int(np.argmax(vs))
            r0, rp, rm = by[fn, L, 0.0], by[fn, L, round(pi, 9)], by[fn, L, round(PSIS[jmin], 9)]
            var = vs[jmax] - vs[jmin]
            K = f(rm, f"cota_{PSI_NAMES[jmax]}") - f(rm, "v_max")
            wp = witness_params(r0)
            ser[fn, L] = dict(var=var, K=K, dv=vs[-1] - vs[0], lo=-(f(rp, "cota_0") - f(rp, "v_max")),
                              hi=f(r0, "cota_pi") - f(r0, "v_max"), gap=f(r0, "gap"), wp=wp)
    par, impar = [4, 6, 8, 10], [5, 7, 9]

    # f = 0: testigo con gap.
    an = max(max(abs(ser["0", L]["wp"][c] - 1) for c in ("cb_rb", "ct_rt", "cr_ar")) for L in range(4, 11))
    num("2e-3", an, "§5: f = 0: máx |c/r - 1| (patas y peldaños, L = 4..10)", "<=")
    for L, t in zip(par, ("1,000", "0,865", "0,819", "0,796")):
        num(t, ser["0", L]["wp"]["jperp"], f"§5: f = 0, L = {L}: J_perp/J")
    lg = [L * ser["0", L]["gap"] for L in par]
    gj = [ser["0", L]["wp"]["gapJ"] for L in par]
    for L, x, t in zip(par, lg, ("0,36", "0,27", "0,22", "0,20")):
        num(t, x, f"§5: f = 0, L = {L}: L gap")
    for L, x, t in zip(par, gj, ("0,82", "0,58", "0,47", "0,42")):
        num(t, x, f"§5: f = 0, L = {L}: gap/J")
    ck.check(all(np.diff(lg) < 0) and all(np.diff(gj) < 0), "§5: f = 0, L par: L gap y gap/J monótonas")
    num("0,18", shanks(lg[-3:]), "§5: f = 0: Shanks de L gap (L = 6, 8, 10)")
    num("0,37", shanks(gj[-3:]), "§5: f = 0: Shanks de gap/J (L = 6, 8, 10)")
    lgi = [L * ser["0", L]["gap"] for L in impar]
    gji = [ser["0", L]["wp"]["gapJ"] for L in impar]
    mono = lambda x: all(np.diff(x) < 0) or all(np.diff(x) > 0)  # noqa: E731
    ck.check(not mono(lgi) and not mono(gji), "§5: f = 0, L impar: L gap y gap/J no monótonas (sin Shanks)")
    ck.check(all(s["hi"] < 0 or s["lo"] > 0 for (fn, L), s in ser.items() if fn == "0"),
             "§5: f = 0: la horquilla fija el signo de v(pi) - v(0) en L = 4..10")
    s10 = ser["0", 10]
    num("-2,929e-3", s10["lo"], "§5: f = 0, L = 10: -R_pi(-pi)")
    num("-2,008e-3", s10["hi"], "§5: f = 0, L = 10: R_0(pi)")
    num("-2,412e-3", s10["dv"], "§5: f = 0, L = 10: v(pi) - v(0)")
    vz = [ser["0", L]["var"] for L in par]
    for L, x, t in zip(par, vz, ("6,9e-2", "1,6e-2", "5,8e-3", "2,4e-3")):
        num(t, x, f"§1, §5: f = 0, L = {L}: variación en psi")
    for (_, p), t in zip(local_exponents(par, vz), ("3,54", "3,64", "3,90")):
        num(t, p, "§5: f = 0: exponente local de la variación")
    rz = [ser["0", L]["var"] / ser["0", L]["K"] for L in par]
    num("0,516", min(rz), "§5: f = 0, L par: variación/cota, el menor")
    num("0,823", max(rz), "§5: f = 0, L par: variación/cota, el mayor")
    ft = {n: (pv, rms) for n, (_, pv), rms in fits(par, vz)}
    num("0,025", ft["L^-p"][1], "§5: f = 0: rms de L^-p")
    num("3,65", ft["L^-p"][0], "§5: f = 0: p de L^-p")
    num("0,051", ft["L^-2 e^(-L/xi)"][1], "§5: f = 0: rms de L^-2 e^(-L/xi)")
    num("3,98", ft["L^-2 e^(-L/xi)"][0], "§5: f = 0: xi de L^-2 e^(-L/xi)")
    num("0,097", ft["L^-1 e^(-L/xi)"][1], "§5: f = 0: rms de L^-1 e^(-L/xi)")
    num("0,14", ft["e^(-L/xi)"][1], "§5: f = 0: rms de e^(-L/xi)")

    # f = pi.
    vp = [ser["pi", L]["var"] for L in par]
    for L, x, t in zip(par, vp, ("5,1e-3", "1,4e-3", "5,2e-4", "1,3e-4")):
        num(t, x, f"§5: f = pi, L = {L}: variación en psi")
    for (_, p), t in zip(local_exponents(par, vp), ("3,21", "3,42", "6,37")):
        num(t, p, "§5: f = pi: exponente local de la variación")
    sg = [int(np.sign(ser["pi", L]["dv"])) for L in par]
    ck.check(sg == [1, -1, -1, 1], f"§5: f = pi, L par: el signo de v(pi) - v(0) alterna ({sg})")
    rp_ = [ser["pi", L]["var"] / ser["pi", L]["K"] for L in par]
    num("0,54", min(rp_), "§5: f = pi, L par: variación/cota, el menor")
    num("0,86", max(rp_), "§5: f = pi, L par: variación/cota, el mayor")
    ft = {n: (pv, rms) for n, (_, pv), rms in fits(par, vp)}
    num("0,090", ft["e^(-L/xi)"][1], "§5: f = pi: rms de e^(-L/xi)")
    num("1,65", ft["e^(-L/xi)"][0], "§5: f = pi: xi de e^(-L/xi)")
    num("0,22", ft["L^-p"][1], "§5: f = pi: rms de L^-p")
    gi = [ser["pi", L]["gap"] for L in impar]
    num("6,87e-3", min(gi), "§5: f = pi, L impar: gap del testigo, el menor")
    num("7,31e-3", max(gi), "§5: f = pi, L impar: gap del testigo, el mayor")
    for L, t in zip(impar, ("2,5e-4", "9,7e-7", "4,5e-7")):
        num(t, ser["pi", L]["var"], f"§5: f = pi, L = {L}: variación en psi")
    v11 = ref("pi", 11)
    num("3,0e-8", max(v11.values()) - min(v11.values()), "§5: f = pi, L = 11: variación en psi")
    ki = [ser["pi", L]["K"] / ser["pi", L]["var"] for L in impar]
    num("10,7", min(ki), "§5: f = pi, L impar: cota/variación, el menor")
    num("16,5", max(ki), "§5: f = pi, L impar: cota/variación, el mayor")


# ---------------------------------------------------------------------------
# §6: diamantes
# ---------------------------------------------------------------------------

def diamantes(ck, num):
    rows = leer("rigidez_diamantes.csv")
    k3 = [r for r in rows if int(r["k"]) == 3]
    k5 = [r for r in rows if int(r["k"]) == 5]
    ck.check(len(k3) == 7 and len(k5) == 7, f"§6: malla de 7 puntos de psi en k = 3 y 5 ({len(k3)}, {len(k5)})")
    c = 3 * (1 / 3 - f(k3[0], "v_cota"))
    num("0,0766202939", c, "§6: c = k(1/3 - v), k = 3 (Proposición 10 de diamantes)")
    e3 = [f(r, "eps_anillo") for r in k3]
    num("-4,3875", e3[0], "§6: k = 3: energía del anillo sin singletes en psi = 0")
    num("-4,3358", e3[-1], "§6: k = 3: energía del anillo sin singletes en psi = pi")
    X3 = [f(r, "X_anillo") for r in k3]
    num("0,509", min(X3), "§6: k = 3: exceso del anillo, el menor")
    num("0,547", max(X3), "§6: k = 3: exceso del anillo, el mayor")
    num("0,230", 3 * c, "§6: 3c")
    num("0,28", min(X3) - 3 * c, "§6, §8: k = 3: margen del exceso sobre 3c")
    dl = max(abs(f(r, "lambda_min") - f(k3[0], "lambda_min")) for r in k3)
    ck.check(dl <= 1e-15, f"§6: k = 3: lambda_min(H_psi) = lambda_min(H_0) a {dl:.1e} (<= 1e-15)")
    num("0,877", min(f(r, "X_anillo") for r in k5), "§6: k = 5: exceso del anillo, al menos", ">=")
    num("1,1e-16", max(float(r["conmutador"]) for r in rows), "§6: máx ||[H_psi, T_m^2]||")
    f0 = {}
    for k in (2, 3, 4):
        f0[k] = [f(r, "v_max") for r in leer(f"diamond_ring_L{k}_igual.csv") if abs(f(r, "f")) < 1e-12]
    ck.check(all(len(v) == 13 for v in f0.values()), "§6: 13 valores de psi en f = 0 (k = 2, 3, 4)")
    num("5,4e-10", max(f0[3]) - f0[3][0], "§6: k = 3: máx v(psi) - v(0)")
    dpar = max(abs(v - 1 / 3) for k in (2, 4) for v in f0[k])
    num("9,9e-9", dpar, "§6: k = 2, 4: máx |v(psi) - 1/3|")
    ck.check(dpar <= 1e-8, f"§6, §8: k par: datos planos a 1e-8 ({dpar:.1e})")


# ---------------------------------------------------------------------------
# §7: qudits
# ---------------------------------------------------------------------------

def limite(ck, num):
    rows = leer("rigidez_qudit_limite.csv")
    inf = {int(r["d"]): r for r in rows if r["n"] == "inf"}
    fin = {(int(r["d"]), int(r["n"])): r for r in rows if r["n"] != "inf"}
    for d, te, tv in ((2, "-0,386294", "0,590863"), (3, "-0,703212", "0,777409"),
                      (4, "-0,825119", "0,860095"), (5, "-0,884730", "0,903941"),
                      (6, "-0,918304", "0,929975")):
        num(te, f(inf[d], "e_inf"), f"§7.2, tabla: e_inf({d})")
        num(tv, f(inf[d], "v_inf"), f"§7.2, tabla: v_inf({d})")
    ck.check(abs(f(inf[2], "v_inf") - (4 * math.log(2) - 1) / 3) < 1e-13, "§7.2: v_inf(2) = (4 ln 2 - 1)/3")
    ck.check(max(n for d, n in fin if d == 2) == 16 and max(n for d, n in fin if d == 3) == 12
             and max(n for d, n in fin if d == 4) == 12, "§7.2: n hasta 16 (d = 2) y 12 (d = 3, 4)")
    ck.check(all(f(fin[4, n], "v_n0") == 1.0 or abs(f(fin[4, n], "v_n0") - 1) < 1e-12 for n in (3, 4)),
             "§7.2: d = 4, n = 3 y 4: v = 1 (n <= d, estado antisimétrico)")
    for (d, n), t in (((3, 6), "0,825694"), ((3, 9), "0,798286"), ((3, 12), "0,789033"),
                      ((4, 8), "0,892101"), ((4, 12), "0,874066")):
        num(t, f(fin[d, n], "v_n0"), f"§7.2: v({n}, 0), d = {d}")
    pr = {k: abs(f(r, "v_primal") - f(r, "v_n0")) for k, r in fin.items() if r["v_primal"] not in ("", "nan")}
    num("3,6e-8", max(v for (d, _), v in pr.items() if d == 3), "§7.2: d = 3, n <= 5: |primal - fórmula|")
    num("1,7e-6", pr[4, 4], "§7.2: d = 4, n = 4: |primal - fórmula|")
    for (d, n), t in (((3, 6), "-2,3177"), ((3, 9), "-2,2547"), ((3, 12), "-2,2318"),
                      ((4, 8), "-2,5605"), ((4, 12), "-2,5147")):
        num(t, f(fin[d, n], "n2_dev"), f"§7.2: n^2(e_P - e_inf), d = {d}, n = {n}")
    num("-2,1932", f(fin[3, 6], "a_cft"), "§7.2: -pi^2(d-1)/(3d), d = 3")
    num("-2,4674", f(fin[4, 8], "a_cft"), "§7.2: -pi^2(d-1)/(3d), d = 4")


def flujo(ck, num):
    rows = leer("rigidez_qudit_flujo.csv")
    malla = {(r["recta"], int(r["n"]), int(r["i_theta"])): r for r in rows if r["tipo"] == "malla"}
    num_hq = max(abs(f(r, "horquilla")) for r in rows)
    ck.check(num_hq < 1e-9, f"§7.3: horquilla de los planos de corte < 1e-9 ({num_hq:.1e})")
    num("6,3e-13", max(abs(f(r, "dif_lanczos")) for r in rows), "§7.3: Lanczos aleatorio, máx |dif|")
    pr = [abs(f(r, "v_primal") - f(r, "v_sup")) for r in rows if r["v_primal"] not in ("", "nan")]
    ck.check(len(pr) == 78, f"§7.3: primal en {len(pr)} puntos de malla (78)")
    num("1,9e-7", max(pr), "§7.3, §8: máx |dual - primal|, n <= 5")
    ina = [r for r in rows if r["v_primal"] not in ("", "nan") and r["estado_primal"] == "optimal_inaccurate"]
    ck.check(len(ina) == 77, f"§7.3, §8: {len(ina)} de los {len(pr)} primales optimal_inaccurate (77 de 78)")
    ck.check(all(r["estado"] == "optimal" for r in rows),
             f"§7.3, §8: planos de corte con estado optimal (no en_caja) en las {len(rows)} filas: caja inactiva")
    cus = [r for r in rows if r["clase"] in ("doblete", "triple")]
    num("0,030", min(f(r, "desdoblamiento") for r in cus), "§7.3: desdoblamiento de corrientes, mayor que", ">=")
    cert = [r for r in cus if r["cert_margen"] not in ("", "nan") and f(r, "cert_margen") > 1e-7]
    nocert = [r for r in cus if r not in cert]
    t0 = [r for r in nocert if r["tipo"] == "malla" and f(r, "theta") == 0 and int(r["n"]) % 3]
    bis = [r for r in nocert if r["tipo"] == "borde"]
    bis_cus = [r for r in cus if r["tipo"] == "borde"]
    indet = [r for r in rows if r["clase"] == "indet"]
    ck.check(len(cus) == 212 and len(cert) == 200 and len(t0) == 8 and len(bis) == 4
             and len(nocert) == 12 and len(bis_cus) == 33 and len(indet) == 8
             and all(r["tipo"] == "borde" for r in indet)
             and sorted({int(r["n"]) for r in t0}) == [4, 5, 7, 8],
             f"§7.3, §8: {len(cus)} filas en cúspide, {len(cert)} certificadas; sin certificar {len(t0)} de "
             f"theta = 0 (n = 4, 5, 7, 8) y {len(bis)} de las {len(bis_cus)} de bisección; {len(indet)} «indet» "
             "(212, 200, 8, 4, 33, 8)")
    v = {}
    for n in range(3, 10):
        v0, vC, vM = (f(malla["raiz", n, i], "v_sup") for i in (0, 8, 12))
        v[n] = (v0, vC, vM)
        ck.check(abs(vM - f(malla["peso", n, 12], "v_sup")) < 1e-7, f"§7.3: n = {n}: M igual en las dos rectas")
    tabla = {3: ("1,000000", "0,584171", "0,600000", "3,74", "3,600"),
             4: ("0,625000", "0,732786", "0,702484", "-1,72", "-1,240"),
             5: ("0,735410", "0,754046", "0,732949", "-0,47", "0,062"),
             6: ("0,825694", "0,746294", "0,754352", "2,86", "2,568"),
             7: ("0,733470", "0,766488", "0,754893", "-1,62", "-1,050"),
             8: ("0,756845", "0,769348", "0,760381", "-0,80", "-0,226"),
             9: ("0,798286", "0,766339", "0,771918", "2,59", "2,136")}
    for n, (a, b, c, dC, dM) in tabla.items():
        v0, vC, vM = v[n]
        num(a, v0, f"§7.3, tabla: v(0), n = {n}")
        num(b, vC, f"§7.3, tabla: v(C), n = {n}")
        num(c, vM, f"§7.3, tabla: v(M), n = {n}")
        num(dC, n * n * (v0 - vC), f"§7.3, tabla: n^2[v(0) - v(C)], n = {n}")
        num(dM, n * n * (v0 - vM), f"§7.3, tabla: n^2[v(0) - v(M)], n = {n}")
    ok = all((v[n][0] > v[n][1]) == (n % 3 == 0) for n in range(3, 10))
    ok2 = all((v[n][0] > v[n][2]) == (n % 3 == 0) for n in range(3, 10) if n != 5) and v[5][0] > v[5][2]
    ck.check(ok and ok2, "§7.3: signo de v(0) - v(C) = + si n = 0 mód 3, - si no (n = 3..9); "
             "v(0) - v(M) igual salvo n = 5")
    # Anclaje en t = 0 en la recta peso: hasta 5L/12 (n = 5), 3L/4 (7), L/2 (8).
    for n, imax in ((5, 5), (7, 9), (8, 6)):
        anc = [abs(f(malla["peso", n, i], "theta_W")) < 1e-6 and malla["peso", n, i]["clase"] == "doblete"
               for i in range(13)]
        ck.check(all(anc[:imax + 1]) and not anc[imax + 1],
                 f"§7.3: n = {n}, recta peso: anclado en t = 0 (doblete) hasta i = {imax}, no en i = {imax + 1}")
    lisos = [r for r in rows if r["clase"] == "liso" and int(r["n"]) % 3]
    son = [r for r in rows if r["tipo"] == "sonda"]
    ck.check(not lisos and len(son) == 3 and all(r["clase"] == "triple" for r in son),
             "§7.3: ningún punto liso con n no múltiplo de 3 (n = 4, 5, 7, 8; sonda n = 10: 3 filas triple)")
    # Bordes de la ventana (bisección), en theta/pi.
    for recta, n, lo, hi in (("raiz", 6, "0,019857", "0,019938"), ("peso", 6, "0,017337", "0,017384"),
                             ("raiz", 9, "0,034587", "0,034668"), ("peso", 9, "0,028614", "0,028661")):
        bs = sorted((f(r, "theta") / pi, r["clase"]) for r in rows
                    if r["recta"] == recta and int(r["n"]) == n and r["tipo"] in ("malla", "borde")
                    and f(r, "theta_sobre_L") <= 1 / 12 + 1e-12)
        li = max(t for t, c in bs if c == "liso")
        cu = min(t for t, c in bs if c != "liso" and t > li)
        num(lo, li, f"§7.3: n = {n}, {recta}: último liso de la bisección (theta/pi)")
        num(hi, cu, f"§7.3: n = {n}, {recta}: primera cúspide de la bisección (theta/pi)")


def asint(ck, num):
    rows = leer("rigidez_qudit_asint.csv")

    def val(seccion, magnitud, **kw):
        sel = [r for r in rows if r["seccion"] == seccion and r["magnitud"] == magnitud
               and all(r[k] == str(v) for k, v in kw.items())]
        assert len(sel) == 1, (seccion, magnitud, kw, len(sel))
        return f(sel[0], "valor"), sel[0]

    num("0,15378", val("teoria", "kappa", d=2)[0], "§7.4: kappa_2 = psi_inf/pi de anillos")
    for punto, r, t in (("C", 0, "2,1932"), ("M", 0, "1,6449"), ("C", 1, "-1,0966"), ("M", 1, "-0,5483")):
        num(t, val("teoria", "coef", d=3, r=r, punto=punto)[0], f"§7.4: coeficiente límite en {punto}, r = {r}")
    for punto, r, t in (("C", 0, "2,193"), ("M", 0, "1,645"), ("C", 1, "-1,097"), ("M", 1, "-0,548")):
        num(t, val("teoria", "coef", d=3, r=r, punto=punto)[0], f"§7.4, tabla: límite en {punto}, r = {r}")
    x, _ = val("teoria", "coef_sin_campo", d=3, r=1, punto="M")
    ck.check(abs(x + pi ** 2 / 6) < 1e-9, f"§7.4: sin campo, M con n no múltiplo de 3: -pi^2/6 ({x:.6f})")
    num("0,14245", val("teoria", "kappa", d=3)[0], "§7.4: kappa_3")
    num("0,12800", val("teoria", "kappa", d=4)[0], "§7.4: kappa_4")
    vr, r_ = val("teoria", "ventana_theta_sobre_pi", d=3, recta="raiz")
    vp, _ = val("teoria", "ventana_theta_sobre_pi", d=3, recta="peso")
    num("0,0950", vr, "§7.4: ventana límite, recta raíz")
    num("0,0822", vp, "§7.4: ventana límite, recta peso")
    num("1,155", vr / vp, "§7.4, §7.4 (n finito): cociente raíz/peso límite (2/sqrt 3)")
    num("0,5717", val("teoria", "anclado_en_0_hasta_theta_sobre_L", recta="peso")[0], "§7.4: peso, anclado en 0 hasta")
    num("0,7142", val("teoria", "anclado_en_M_desde_theta_sobre_L", recta="peso")[0], "§7.4: peso, anclado en M desde")
    num("0,2858", val("teoria", "anclado_en_0_hasta_theta_sobre_L", recta="raiz")[0], "§7.4: raíz, anclado en 0 hasta")
    num("0,6192", val("teoria", "meseta_L/3_hasta_theta_sobre_L", recta="raiz")[0], "§7.4: raíz, meseta L/3 hasta")

    # Energía sin giro frente a (1).
    e = {int(r["n"]): f(r, "valor") for r in rows if r["seccion"] == "energia" and r["magnitud"] == "n(E0-n*einf)"}
    num("-2,2318", e[12], "§7.2, §7.4: n(E_0 - n e_inf), n = 12")
    for (a, b), t in zip(((4, 5), (7, 8), (10, 11)), ("2,33", "2,31", "2,30")):
        num(t, (e[a] + e[b]) / 2, f"§7.4: media de n(E_0 - n e_inf) en n = {a}, {b}")

    # Tabla de coeficientes.
    co = {}
    for r in rows:
        if r["seccion"] == "coef" and r["magnitud"] == "campo":
            co[int(r["n"]), r["punto"]] = (f(r, "valor"), f(r, "tarea7"))
    sonda = [f(r, "valor") for r in rows if r["magnitud"] == "tarea7_sonda"][0]
    tab = {3: ("2,853", "3,742", "2,250", "3,600"), 6: ("2,500", "2,858", "1,915", "2,568"),
           9: ("2,427", "2,588", "1,846", "2,136"), 12: ("2,396", None, "1,817", None),
           4: ("-2,018", "-1,725", "-1,657", "-1,240"), 7: ("-1,708", "-1,618", "-1,098", "-1,050"),
           10: ("-1,564", None, "-0,959", "-0,936"), 5: ("-0,560", "-0,466", "0,043", "0,062"),
           8: ("-0,835", "-0,800", "-0,231", "-0,226"), 11: ("-0,943", None, "-0,341", None)}
    for n, (cf, c7, mf, m7) in tab.items():
        num(cf, co[n, "C"][0], f"§7.4, tabla: familia, C, n = {n}")
        num(mf, co[n, "M"][0], f"§7.4, tabla: familia, M, n = {n}")
        if c7:
            num(c7, co[n, "C"][1], f"§7.4, tabla: óptimo, C, n = {n}")
        if m7:
            num(m7, co[n, "M"][1] if n != 10 else sonda, f"§7.4, tabla: óptimo, M, n = {n}")
    lim = {(p, r): val("teoria", "coef", d=3, r=r, punto=p)[0] for p in "CM" for r in (0, 1)}
    mono = True
    for p in "CM":
        for r0 in (0, 1, 2):
            ns = [n for n in range(3, 13) if n % 3 == r0]
            xs = [co[n, p][0] - lim[p, min(r0, 1)] for n in ns]
            mono &= all(np.sign(xs) == np.sign(xs[0])) and all(np.diff(np.abs(xs)) < 0)
    ck.check(mono, "§7.4: las seis sucesiones de la familia son monótonas hacia su límite y no lo cruzan")
    num("1,092", co[12, "C"][0] / lim["C", 0], "§7.4: n = 12, C: cociente con el límite")
    num("1,104", co[12, "M"][0] / lim["M", 0], "§7.4: n = 12, M: cociente con el límite")
    med = {(int(r["n"].split("-")[0]), r["punto"]): f(r, "valor") for r in rows if r["magnitud"] == "media_r1_r2"}
    for (a, p), t in (((4, "C"), "-1,289"), ((7, "C"), "-1,272"), ((10, "C"), "-1,253"),
                      ((4, "M"), "-0,807"), ((7, "M"), "-0,664"), ((10, "M"), "-0,650")):
        num(t, med[a, p], f"§7.4: media de los pares desde n = {a}, {p}")
    iso = {(int(r["n"]), r["punto"]): f(r, "valor") for r in rows if r["seccion"] == "coef" and r["magnitud"] == "iso"}
    m7 = {n: (co[n, "M"][1] if n != 10 else sonda) for n in (7, 8, 10)}
    ck.check(all(abs(m7[n] - co[n, "M"][0]) < 0.05 and abs(m7[n] - iso[n, "M"]) > 1 for n in m7),
             "§7.4: M, n = 7, 8, 10: el óptimo a menos de 0,05 de la familia con campo y a más de 1 de la "
             "familia sin campo: " + ", ".join(f"n = {n}: {m7[n] - co[n, 'M'][0]:+.3f}, {m7[n] - iso[n, 'M']:+.3f}"
                                               for n in m7))
    an ={(n, p): co[n, p][1] - iso[n, p] for n in (6, 9) for p in "CM"}
    for (n, p), t in (((6, "C"), "0,358"), ((9, "C"), "0,161"), ((6, "M"), "0,653"), ((9, "M"), "0,290")):
        num(t, an[n, p], f"§7.4: anisotropía de dos sitios (óptimo - familia), {p}, n = {n}")
    num("0,45", an[9, "C"] / an[6, "C"], "§7.4: cociente de la anisotropía, C (n = 9/6)")
    num("0,44", an[9, "M"] / an[6, "M"], "§7.4: cociente de la anisotropía, M (n = 9/6)")
    ck.check(abs(an[9, "M"] / an[6, "M"] - (6 / 9) ** 2) < 0.01 and abs(an[9, "C"] / an[6, "C"] - (6 / 9) ** 2) < 0.01,
             "§7.4: los cocientes están cerca de (6/9)^2 = 0,444 (a 0,01)")

    # Rigidez local y ventanas.
    bl = {(int(r["n"]), r["recta"]): f(r, "valor") for r in rows
          if r["seccion"] == "rigidez" and r["magnitud"] == "B_local" and r["punto"] == "0"}
    ck.check(all(abs(bl[n, "raiz"] - bl[n, "peso"]) < 1e-6 for n in (3, 6, 9, 12)),
             "§7.4: rigidez local en 0 isótropa: raíz = peso a 1e-6")
    for n, t, q in ((6, "0,4033", "1,21"), (9, "0,3834", "1,15"), (12, "0,3750", "1,125")):
        num(t, bl[n, "raiz"], f"§7.4: rigidez local en 0, n = {n}")
        num(q, 3 * bl[n, "raiz"], f"§7.4: rigidez local en 0 / (1/3), n = {n}")
    fr = {(int(r["n"]), r["recta"]): f(r, "valor") for r in rows if r["magnitud"] == "fraccion_tarea7"}
    for (n, rc), t in (((6, "raiz"), "0,210"), ((6, "peso"), "0,211"), ((9, "raiz"), "0,365"), ((9, "peso"), "0,348")):
        num(t, fr[n, rc], f"§7.4: ventana del óptimo / límite, {rc}, n = {n}")
    cq = {int(r["n"]): f(r, "valor") for r in rows if r["magnitud"] == "cociente_raiz_peso_tarea7"}
    num("1,146", cq[6], "§7.4: cociente raíz/peso de la ventana del óptimo, n = 6")
    num("1,209", cq[9], "§7.4: cociente raíz/peso de la ventana del óptimo, n = 9")
    vi = {(int(r["n"]), r["recta"]): f(r, "valor") for r in rows if r["magnitud"] == "iso_theta_sobre_pi"}
    for (n, rc), t in (((6, "raiz"), "0,0649"), ((9, "raiz"), "0,0674"), ((12, "raiz"), "0,0682"),
                       ((6, "peso"), "0,0328"), ((9, "peso"), "0,0405"), ((12, "peso"), "0,0441")):
        num(t, vi[n, rc], f"§7.4: ventana de la familia sin anisotropía, {rc}, n = {n}")
    ck.check(round(100 * vi[12, "raiz"] / vr) == 72 and round(100 * vi[12, "peso"] / vp) == 54,
             f"§7.4: n = 12: {100 * vi[12, 'raiz'] / vr:.1f} % y {100 * vi[12, 'peso'] / vp:.1f} % del límite (72 y 54)")
    for rc in ("raiz", "peso"):
        dd = np.diff([vi[n, rc] for n in (6, 9, 12)])
        ck.check(dd[0] > dd[1] > 0, f"§7.4: ventana de la familia, {rc}: incrementos que se reducen {dd}")
    x, _ = val("proposicion", "filas_que_cumplen")
    ck.check(x == 107, f"§7.4: la Proposición de varios sectores se cumple en {x:.0f} óptimos (107)")


# ---------------------------------------------------------------------------
# §9: tiempos y memoria (README)
# ---------------------------------------------------------------------------

def medidas(ck, s, vistas):
    readme = README.read_text()
    for art, rd in MEDIDAS:
        ok = rd in readme and (s is None or art in s)
        ck.check(ok, f"§9: «{art}» en el artículo y «{rd}» en el README")
        vistas.update(numeros(art))


def main():
    ck = Checker("rigidez_numeros")
    ruta = Path(sys.argv[1]) if len(sys.argv) > 1 else TEX
    s = cuerpo(ruta.read_text()) if ruta.exists() else None
    if s is None:
        ck.info(f"no está {ruta}: solo se comprueban las cifras contra los CSV")
    num = Cifras(ck, s)
    anillos(ck, num)
    escaleras(ck, num)
    diamantes(ck, num)
    limite(ck, num)
    flujo(ck, num)
    asint(ck, num)
    medidas(ck, s, num.vistas)
    if s is not None:
        todos = numeros(s)
        sobran = sorted(set(todos) - num.vistas, key=lambda t: todos.index(t))
        ck.check(not sobran, f"ningún número sin guion: {len(set(todos))} números distintos en el .tex, "
                             f"{len(sobran)} sin comprobar" + (f": {', '.join(sobran)}" if sobran else ""))
    ck.finish()


if __name__ == "__main__":
    main()
