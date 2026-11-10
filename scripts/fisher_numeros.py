"""Números del artículo de Fisher (solo lectura).

Artículo (~/fisica/holonomia_fisher.tex, «Holonomía y pegado de perspectivas de Werner II: información de
Fisher, primera ley del entrelazamiento y el testigo como hamiltoniano modular»): todas las secciones. No
resuelve ningún SDP ni diagonaliza nada: lee los CSV de fisher_kubo_mori.py, anillo_entropia.py,
fisher_energia_libre.py, anillo_termico.py y regimen_degenerado.py.

Dos comprobaciones:
  1. Cada cifra contra su CSV, redondeada como en el texto (cifra() de diamond_numeros.py). Las
     diferencias relativas se recalculan a partir de las columnas del CSV.
  2. Cobertura: todo número del texto (decimales y m\\times10^{e}, extraídos como en rigidez_numeros.py)
     está comprobado en 1 o en OTROS, cada uno con su motivo.

Uso: python scripts/fisher_numeros.py [ruta del .tex]
     (por defecto ~/fisica/holonomia_fisher.tex; menos de 1 s)
"""
import _common  # noqa: F401  (fija OMP_NUM_THREADS antes de numpy)
from _common import Checker

import sys
from pathlib import Path

import numpy as np

from diamond_numeros import cifra
from rigidez_numeros import cuerpo, leer, numeros, sin_signo

TEX = Path.home() / "fisica" / "holonomia_fisher.tex"

# Números del texto que no salen de los CSV de fisher_kubo_mori.py, con su motivo.
OTROS = {
    "2,773": "ln 16",
    "1e-12": "tolerancia de Newton de MaxEnt.resuelve y cota de los residuos",
    "1e-10": "tolerancia de las comprobaciones de la primera ley y de N_h(W*[h]) en fisher_kubo_mori.py",
    "1e-8": "tolerancia de la comparación de S0 con el segundo código en fisher_kubo_mori.py",
    "1e-6": "cota de la diferencia relativa de beta = -dS/dv (salida del guion: 1,3e-7 y 5e-8) y del rango de la curvatura",
    "1e-9": "rango de la curvatura (dif. rel. mínima 4,7e-9)",
    "1e-15": "la fórmula en la dirección de gauge (columna S2_kubo_mori: 0 y -1,3e-15) y la alineación x.O = beta W "
             "(salida de fisher_kubo_mori.py: 1 - coseno <= 2,3e-15)",
    "2e-3": "paso h de las diferencias finitas de cinco puntos (H en fisher_kubo_mori.py)",
    "4e-2": "paso h del SDP reoptimizado",
    "2e-2": "paso h del SDP reoptimizado",
    "1e-2": "paso h del SDP reoptimizado y eps de la tabla de regímenes",
    "1e-3": "eps",
    "3e-4": "eps",
    "1e-4": "eps",
    "0,3": "eps y v",
    "0,1": "v",
    "0,2": "v",
    "0,005": "Theta/pi de la tabla del anillo",
    "0,015": "Theta/pi de la tabla del anillo",
    "0,0195": "Theta/pi de la tabla del anillo",
    "0,05": "Theta/pi de la tabla del anillo",
    "0,25": "Theta/pi de la tabla del anillo",
    "0,5": "Theta/pi de la tabla del anillo",
    "0,75": "Theta/pi de la tabla del anillo",
    "0,01": "Theta/pi de la tabla del anillo",
    "0,018587": "anchura de la ventana, n = 6, tabla 1 del primer artículo (edge.py la comprueba)",
    "2e-5": "error del contraste de la fórmula de los pesos con el SDP en el primer artículo (observación tras el Teorema 19)",
    "1e-5": "1 - v/vmax del límite de S_G (anillo_entropia.py)",
    "3e-3": "distancia al borde, en pi (CERCA de anillo_entropia.py)",
    "5e-3": "paso h del SDP en vmax'' (fisher_energia_libre.py)",
    "0,4": "v de los anillos (VS_ESCALA de anillo_termico.py)",
    "0,55": "v de los anillos (VS_ESCALA de anillo_termico.py)",
    "0,185": "a del testigo de los diamantes con k par (regimen_degenerado.py)",
}


def main():
    chk = Checker("fisher_numeros")
    ruta = Path(sys.argv[1]) if len(sys.argv) > 1 else TEX
    s = cuerpo(ruta.read_text())
    en_tex = set(numeros(s))
    vistas = set()

    def c(txt, x, msg):
        key = sin_signo(txt)
        vistas.add(key)
        ok = cifra(abs(x), key)
        nota = "" if key in en_tex else "  [no aparece en el .tex]"
        chk.check(ok and not nota, f"{msg}: {x:.6e} -> «{txt}»{nota}")

    # §3, tabla de la curvatura y texto
    cur = {(int(r["L"]), r["direccion"], float(r["v"])): r for r in leer("fisher_kubo_mori.csv")}
    tabla = {
        (3, 0.1): ("0,023582", "0,031121", "0,0075389", "4,0e-8"),
        (3, 0.2): ("0,081836", "0,14356", "0,061722", "1,2e-8"),
        (3, 0.3): ("0,18411", "0,54343", "0,35932", "4,7e-9"),
        (4, 0.1): ("0,039042", "0,037030", "0,0020117", "2,7e-7"),
        (4, 0.2): ("0,17409", "0,14266", "0,031430", "1,5e-8"),
        (4, 0.3): ("0,50154", "0,32168", "0,17986", "1,3e-8"),
    }
    for (L, v), (fis, mul, s2, rel) in tabla.items():
        r = cur[L, "plaquetas", v]
        km, fd = float(r["S2_kubo_mori"]), float(r["S2_fd"])
        c(fis, float(r["fisher"]), f"2x{L}, v = {v}: Fisher")
        c(mul, float(r["multiplicadores"]), f"2x{L}, v = {v}: multiplicadores")
        c(s2, km, f"2x{L}, v = {v}: S''")
        c(rel, abs(fd - km) / abs(km), f"2x{L}, v = {v}: dif. rel. con las diferencias finitas")
    for L, txt in ((3, "4,0e-8"), (4, "1,2e-6")):
        r = cur[L, "azar", 0.2]
        km, fd = float(r["S2_kubo_mori"]), float(r["S2_fd"])
        c(txt, abs(fd - km) / abs(km), f"2x{L}: dif. rel. en la dirección al azar")
    for L, txt in ((3, "5,6e-11"), (4, "4,9e-8")):
        c(txt, float(cur[L, "gauge", 0.2]["S2_fd"]), f"2x{L}: diferencias finitas en la dirección de gauge")
    rels = {L: [abs(float(r["pred_dS_F_pi8"]) - float(r["dS_csv"])) / abs(float(r["dS_csv"]))
                for (LL, d, v), r in cur.items() if LL == L and d == "plaquetas"] for L in (3, 4)}
    c("3,4e-3", min(rels[3]), "2x3: menor dif. rel. de 1/2 S'' t^2 con Delta S_hol (F = pi/8)")
    c("9,6e-3", max(rels[3]), "2x3: mayor dif. rel.")
    c("2,1e-3", min(rels[4]), "2x4: menor dif. rel.")
    c("2,4e-3", max(rels[4]), "2x4: mayor dif. rel.")

    # §4, tabla de la primera ley
    pl = {(int(r["L"]), round(float(r["F"]) / np.pi, 6)): r for r in leer("fisher_kubo_mori_primera_ley.csv")}
    tabla = {
        (3, 0.125): ("5,2668e-4", "1,2295e-3", "1,2299e-3", "7,028e-4", "7,011e-4"),
        (3, 0.5): ("7,9377e-3", "1,9567e-2", "1,9679e-2", "1,1629e-2", "1,1218e-2"),
        (3, 1.0): ("2,6235e-2", "7,693e-2", "7,871e-2", "5,070e-2", "4,487e-2"),
        (3, 2.0): ("4,6992e-2", "0,2871", "0,3149", "0,2401", "0,1795"),
        (4, 0.125): ("1,5111e-4", "6,874e-4", "6,875e-4", "8,385e-4", "8,390e-4"),
        (4, 0.5): ("2,3337e-3", "1,0965e-2", "1,1000e-2", "1,3298e-2", "1,3424e-2"),
        (4, 1.0): ("8,3315e-3", "4,344e-2", "4,400e-2", "5,177e-2", "5,369e-2"),
        (4, 2.0): ("2,1121e-2", "0,1671", "0,1760", "0,1883", "0,2148"),
    }
    for (L, F), txts in tabla.items():
        r = pl[L, F]
        cols = ("dS", "dK", "dK_2orden", "S_rel", "S_rel_fisher")
        for txt, col in zip(txts, cols, strict=True):
            c(txt, float(r[col]), f"2x{L}, F = {F} pi: {col}")
        chk.check(abs(float(r["dS"]) - (float(r["dK"]) - float(r["S_rel"]))) < 1e-10,
                  f"2x{L}, F = {F} pi: Delta S = Delta<K> - S_rel a 1e-10")

    # §5, alineación y regímenes
    vm = {(int(r["L"]), float(r["eps"])): r for r in leer("fisher_kubo_mori_vmax.csv")}
    res = {r["magnitud"]: float(r["valor"]) for r in leer("fisher_kubo_mori_resumen.csv")}
    c("0,99986", res["cos_sdp_simetrico_4"], "2x4: coseno entre el testigo del SDP y el simétrico")
    t0 = {(int(r["L"]), float(r["eps"])): r for r in leer("fisher_kubo_mori_T0.csv")}
    c("0,535565486", float(t0[4, 0.1]["vmax"]), "2x4: lambda_max del testigo simétrico")
    chk.check(all(float(r["residuo"]) <= 1e-12 for r in t0.values()), "residuos de la tabla de regímenes <= 1e-12")
    c("0,091121", float(t0[4, 0.1]["gap"]), "2x4: gap del testigo simétrico")
    c("0,091121", res["gap_simetrico_4"], "2x4: gap del testigo simétrico (resumen)")
    chk.check(abs(float(t0[3, 0.1]["gap"]) - 1 / 3) < 1e-12 and abs(res["gap_simetrico_3"] - 1 / 3) < 1e-12,
              "2x3: gap 1/3 de los triángulos desacoplados")
    chk.check(int(float(t0[3, 0.1]["degeneracion"])) == 16 and int(float(t0[4, 0.1]["degeneracion"])) == 1,
              "degeneraciones 16 (2x3) y 1 (2x4)")
    chk.check(abs(float(t0[3, 0.1]["vmax"]) - 1 / 3) < 1e-12, "2x3: vmax = 1/3 (Proposiciones 8 y 9 del primer artículo)")
    for eps, txt in ((1e-2, "0,220"), (1e-3, "0,0722"), (1e-4, "0,0231")):
        c(txt, res[f"peldano_pata_{eps:.0e}".replace("e-0", "e-")], f"2x3: cociente peldaño/pata en {eps:g}")
    for eps, txt in ((1e-3, "0,6781"), (3e-4, "0,6742"), (1e-4, "0,6731")):
        c(txt, float(t0[4, eps]["beta_gap_menos_ln"]), f"2x4: beta gap - ln(1/eps) en {eps:g}")
    c("3,162", float(t0[3, 1e-4]["beta"]) / float(t0[3, 1e-3]["beta"]), "2x3: beta(1e-4)/beta(1e-3)")
    c("1,797", float(t0[3, 1e-4]["S"]), "2x3: S en 1e-4 (texto)")
    tabla = {
        (4, 1e-2): ("42,740", "0,2886", "33,718", "5,2964", "0,789"),
        (4, 1e-3): ("68,367", "0,04251", "72,251", "5,2475", "1,057"),
        (4, 3e-4): ("81,622", "0,01488", "88,560", "5,1969", "1,085"),
        (4, 1e-4): ("93,691", "0,00561", "102,435", "5,1803", "1,093"),
        (3, 1e-2): ("46,024", "2,0763", "0,2790", "0,2634", None),
        (3, 1e-3): ("147,68", "1,8645", "0,3210", "0,2942", None),
        (3, 1e-4): ("466,96", "1,7972", "0,33605", "0,3040", None),
    }
    for (L, eps), (b, S, km, sld, q) in tabla.items():
        r = t0[L, eps]
        c(b, float(r["beta"]), f"2x{L}, eps = {eps:g}: beta")
        c(S, float(r["S"]), f"2x{L}, eps = {eps:g}: S")
        c(km, float(r["fisher_kubo_mori"]), f"2x{L}, eps = {eps:g}: Fisher de Kubo-Mori")
        c(sld, float(r["fisher_sld"]), f"2x{L}, eps = {eps:g}: SLD")
        if q:
            c(q, float(r["fisher_kubo_mori"]) / float(r["beta"]), f"2x{L}, eps = {eps:g}: KM/beta")
    peso = [1 - float(t0[4, e]["peso_max"]) for e in (1e-2, 1e-3, 1e-4)]
    chk.check(all(0.5 < p / e / 5.88 < 1.5 for p, e in zip(peso, (1e-2, 1e-3, 1e-4))),
              f"2x4: el peso fuera del fundamental cae como eps ({', '.join(f'{p:.2e}' for p in peso)})")

    # §6, susceptibilidad de fidelidad
    ch = leer("fisher_kubo_mori_chiF.csv")
    re_ = {float(r["h"]): float(r["valor"]) for r in ch if r["familia"] == "reoptimizado"}
    for h, txt in ((4e-2, "4,7435"), (2e-2, "5,0496"), (1e-2, "5,1388")):
        c(txt, re_[h], f"4 chi_F reoptimizado, h = {h:g}")
    c("5,1697", res["chiF_reoptimizado_richardson"], "4 chi_F reoptimizado, Richardson")
    c("5,1685", (4 * re_[1e-2] - re_[2e-2]) / 3, "4 chi_F reoptimizado, una etapa de Richardson")
    c("0,01523", res["chiF_girado_sdp"], "4 chi_F del testigo girado")
    c("0,01520", next(float(r["valor"]) for r in ch if r["familia"] == "girado_maxima_entropia"),
      "4 chi_F del testigo de máxima entropía girado")
    sl = {float(r["eps"]): float(r["valor"]) for r in ch if r["familia"] == "sld_maxima_entropia"}
    for eps, txt in ((1e-3, "5,2475"), (3e-4, "5,1969"), (1e-4, "5,1803")):
        c(txt, sl[eps], f"SLD en eps = {eps:g}")
    c("5,1720", res["sld_limite_aitken"], "límite de la SLD (Aitken)")
    c("4,6e-4", abs(res["sld_limite_aitken"] - res["chiF_reoptimizado_richardson"]) / res["chiF_reoptimizado_richardson"],
      "dif. rel. SLD frente a reoptimizado")
    chk.check(round(res["chiF_reoptimizado_richardson"] / res["chiF_girado_sdp"]) == 339 and "339" in s,
              f"cociente reoptimizado/girado = {res['chiF_reoptimizado_richardson'] / res['chiF_girado_sdp']:.2f} -> 339")
    d1, d2 = sl[1e-3] - sl[3e-4], sl[3e-4] - sl[1e-4]
    chk.check(round(d1 / d2) == 3, f"las diferencias de la SLD caen un factor {d1 / d2:.2f} -> 3")
    fx = {float(r["eps"]): float(r["valor"]) for r in ch if r["familia"] == "fundamental_x"}
    c("4,8048", fx[1e-3], "4 chi_F del fundamental de x(t).O, eps = 1e-3")
    c("5,1334", fx[1e-4], "4 chi_F del fundamental de x(t).O, eps = 1e-4")
    chk.check(int(res["k_efectivo"]) == 17, f"k efectivo = {res['k_efectivo']:.0f}")
    c("1,29e-4", res["infidelidad_sin_girar"], "infidelidad sin girar")
    c("5,04e-6", res["infidelidad_giro_efectivo"], "infidelidad con el giro efectivo")
    chk.check(round(100 * (1 - res["infidelidad_giro_efectivo"] / res["infidelidad_sin_girar"])) == 96,
              "el giro efectivo explica el 96 % de la infidelidad")
    c("5,17e-5", res["vmax_h_menos_vmax0"], "vmax(+-1e-2) - vmax(0)")
    c("5,17", res["chiF_reoptimizado_richardson"], "5,17 en el cociente de la raíz")
    c("0,0152", res["chiF_girado_sdp"], "0,0152 en el cociente de la raíz")

    # §5, anillo de 6 (anillo_entropia.py)
    an = {round(float(r["Theta_pi"]), 7): r for r in leer("anillo_entropia.csv")}
    filas = sorted(an.values(), key=lambda r: float(r["Theta_pi"]))
    borde = next(float(r["Theta_pi"]) for r in filas if int(r["en_cuspide"])) - 1e-4
    c("0,0185865", borde, "anillo 6: borde de la ventana (primer punto de la cúspide menos 1e-4)")
    chk.check(all(int(r["dim_autoespacio"]) == 1 and float(r["S"]) < 1e-12 for r in filas if float(r["Theta_pi"]) < borde)
              and all(int(r["dim_autoespacio"]) == 2 and int(r["rango"]) == 4 for r in filas if float(r["Theta_pi"]) > borde),
              "anillo 6: simple y S = 0 dentro; doblete con rango 4/4 fuera")
    c("1,6e-8", max(float(r["residuo"]) for r in filas), "anillo 6: residuo máximo del primal")
    tabla = {0.0195: ("5,772e-4", "0,004882"), 0.05: ("0,019635", "0,09662"), 0.1: ("0,050025", "0,19859"),
             0.25: ("0,13577", "0,39721"), 0.5: ("0,26543", "0,57866"), 0.75: ("0,38500", "0,66646")}
    for t, (pm, S) in tabla.items():
        c(pm, float(an[t]["p_menos"]), f"anillo 6, Theta = {t} pi: p_-")
        c(S, float(an[t]["S"]), f"anillo 6, Theta = {t} pi: S")
    chk.check(abs(float(an[1.0]["p_menos"]) - 0.5) < 1e-12 and abs(float(an[1.0]["S"]) - np.log(2)) < 1e-12,
              "anillo 6, Theta = pi: p_- = 1/2, S = ln 2")
    err = max(abs(float(an[t]["p_menos"]) - float(an[t]["p_menos_pesos"])) for t in (0.1, 0.25, 0.5, 0.75, 1.0))
    chk.check(1e-9 < err < 5e-9, f"anillo 6: p_- frente a la fórmula de los pesos, error {err:.1e} -> «3e-9»")
    vistas.add("3e-9")
    cerca = [r for r in filas if any(abs(float(r["Theta_pi"]) - borde - d) < 1e-9 for d in (1e-4, 3e-4, 1e-3, 3e-3))]
    chk.check(len(cerca) == 4, "anillo 6: los cuatro puntos junto al borde (1e-4, 3e-4, 1e-3, 3e-3)")
    pend = [float(r["p_menos"]) / (float(r["Theta_pi"]) - borde) for r in cerca]
    c("0,632", max(pend), "anillo 6: pendiente máxima de p_- cerca del borde")
    c("0,631", min(pend), "anillo 6: pendiente mínima")
    for r, txt in zip(cerca, ("1,00000", "0,99999", "0,99996", "0,99987")):
        pm = float(r["p_menos"])
        c(txt, float(r["S"]) / (pm * (1 - np.log(pm))), f"anillo 6: S/[p(1 - ln p)] en {float(r['Theta_pi']) - borde:.0e}")
    lim = {float(r["Theta_pi"]): r for r in leer("anillo_entropia_limite.csv")}
    for t, txt in ((0.01, "0,00078"), (0.25, "0,39920"), (1.0, "0,69371")):
        c(txt, float(lim[t]["S_G"]), f"anillo 6: S_G en 1 - v/vmax = 1e-5, Theta = {t} pi")

    # Energía libre modular (fisher_energia_libre.py)
    leg = leer("fisher_energia_libre.csv")
    con_S = [r for r in leg if r["S_G"] == r["S_G"] and r["S_G"] != "nan"]
    c("4,4e-15", max(abs(float(r["min_dual"]) - float(r["S_G"])) for r in con_S), "Legendre: min_beta [Phi - beta v] - S_G")
    c("4,3e-8", max(abs(float(r["argmin"]) / float(r["beta_star"]) - 1) for r in con_S), "Legendre: argmin frente a x.t1")
    rejilla = {int(r["L"]): r for r in leg if r["S_G"] == "nan" and abs(float(r["beta_star"]) - 300) < 1e-9}
    d4 = float(rejilla[4]["Phi"]) - 300 * float(t0[4, 0.1]["vmax"])
    chk.check(abs(d4) < 1e-11, f"2x4: Phi - beta vmax = {d4:.1e} en beta = 300 -> «1e-11» (redondeo de Phi ~ 160)")
    vistas.add("1e-11")
    c("1,7903", float(rejilla[3]["Phi"]) - 300 / 3, "2x3: Phi - beta vmax en beta = 300")
    cb = leer("fisher_energia_libre_curvatura.csv")
    c("6e-11", max(abs(float(r["Phi_tt"]) / float(r["S2"]) - 1) for r in cb), "S'' = Phi_tt(beta*): dif. rel. máxima")
    ko = leer("fisher_energia_libre_kohn.csv")
    for L, lo, hi in ((4, "1,05", "1,60"), (3, "0,21", "0,76")):
        q = [float(r["dS"]) / float(r["cota"]) for r in ko if int(r["L"]) == L]
        chk.check(len(q) == 12 and (L == 3 or all(float(r["cota"]) < 0 for r in ko if int(r["L"]) == L)),
                  f"2x{L}: doce casos de la cota de Kohn térmica" + (", todos con cota negativa" if L == 4 else ""))
        c(lo, min(q), f"2x{L}: menor Delta S / cota")
        c(hi, max(q), f"2x{L}: mayor Delta S / cota")
    fl = {(int(r["L"]), round(float(r["beta"]))): r for r in leer("fisher_energia_libre_T0.csv")}
    fe = {(int(r["L"]), float(r["eps"])): r for r in leer("fisher_energia_libre_T0.csv")}
    for eps, txt in ((1e-2, "0,7300"), (1e-3, "0,9974"), (1e-4, "1,0338")):
        c(txt, float(fe[4, eps]["f_tt"]), f"2x4: f_beta''(0) en eps = {eps:.0e}")
    for b in (200, 300):
        c("1,038025", float(fl[4, b]["f_tt"]), f"2x4: f_beta''(0) en beta = {b}")
    fr = {r["magnitud"]: float(r["valor"]) for r in leer("fisher_energia_libre_resumen.csv")}
    c("1,03804", fr["vmax_tt_4"], "2x4: vmax''(0) del SDP")
    c("1,7e-5", fr["f_tt_lim_4"] - fr["vmax_tt_4"], "2x4: f_beta'' límite menos vmax''")
    c("0,05627", fr["kohn_tt_4"], "2x4: curvatura del testigo girado")
    c("18,4", fr["vmax_tt_4"] / fr["kohn_tt_4"], "2x4: cociente de curvaturas reoptimizada/girada")
    for eps, txt in ((1e-1, "0,0260"), (1e-2, "0,0435"), (1e-3, "0,0514"), (1e-4, "0,0542")):
        c(txt, float(fe[3, eps]["f_tt"]), f"2x3: f_beta''(0) en eps = {eps:.0e}")

    # Anillos térmicos (anillo_termico.py)
    c("0,59086", (4 * np.log(2) - 1) / 3, "vmax del anillo infinito, (4 ln 2 - 1)/3")
    esc = {(int(r["n"]), float(r["v"])): r for r in leer("anillo_termico_escala.csv")}
    c("0,46847962", float(esc[16, 0.4]["S_G"]) / 16, "anillo 16, v = 0,4: S_G/n")
    c("0,46847959", float(esc[18, 0.4]["S_G"]) / 18, "anillo 18, v = 0,4: S_G/n")
    esca = {r["direccion"] + r["L"]: r for r in leer("anillo_termico_escaleras.csv")}
    chk.check(max(abs(float(esca[f"plaquetas{L}"]["PQ_d2"]) - 1) for L in (3, 4)) <= 7e-9,
              "escaleras, plaquetas: |PQ/d^2 - 1| <= 7e-9 -> «7e-9»")
    vistas.add("7e-9")
    c("1,0193", float(esca["azar3"]["PQ_d2"]), "2x3, dirección al azar: PQ/d^2")
    c("1,0080", float(esca["azar4"]["PQ_d2"]), "2x4, dirección al azar: PQ/d^2")
    an8 = {round(float(r["Theta"]) / np.pi, 6): float(r["dS"]) for r in leer("anillo_termico.csv")
           if int(r["n"]) == 8 and abs(float(r["v_frac"]) - 0.5) < 1e-9}
    c("2,00005", an8[1.0] / an8[0.5], "anillo 8, v = 0,5 vmax: Delta S(pi)/Delta S(pi/2)")
    xi = {(float(r["v"]), r["paridad"]): r for r in leer("anillo_termico_xi.csv")}
    tabla = {0.2: ("0,1815", "0,5937", "0,5946", "3,27"), 0.3: ("0,2777", "0,7821", "0,7820", "2,82"),
             0.4: ("0,4016", "1,0401", "1,0401", "2,59"), 0.5: ("0,6203", "1,5254", "1,5252", "2,46"),
             0.55: ("0,8997", "2,168", "2,152", "2,41")}
    for v, (tb, xp, xim, cociente) in tabla.items():
        p, im = xi[v, "pares"], xi[v, "impares"]
        c(tb, float(p["b"]), f"anillos, v = {v}: b del mayor anillo par")
        c(xp, float(p["xi"]), f"anillos, v = {v}: xi de los pares")
        c(xim, float(im["xi"]), f"anillos, v = {v}: xi de los impares")
        c(cociente, float(p["xi_sobre_b"]), f"anillos, v = {v}: xi/b")
    c("2,4", float(xi[0.55, "pares"]["xi_sobre_b"]), "xi ~ 2,4 b (el mayor v)")
    c("0,3", 100 * abs(float(xi[0.55, "impares"]["b"]) / float(xi[0.55, "pares"]["b"]) - 1),
      "v = 0,55: deriva de b entre los anillos usados, en %")
    c("0,7", 100 * abs(float(xi[0.55, "pares"]["xi"]) / float(xi[0.55, "impares"]["xi"]) - 1),
      "v = 0,55: diferencia de xi entre pares e impares, en %")

    # Régimen degenerado (regimen_degenerado.py)
    rd = {r["magnitud"]: float(r["valor"]) for r in leer("regimen_degenerado_resumen.csv")}
    c("0,6028199", rd["a_prisma"], "2x3: a")
    c("1,7660521", rd["S_prisma"], "2x3: S*")
    c("1,110174", rd["c2_prisma"], "2x3: c_2")
    c("4,66745", rd["pred_beta_sqrt_eps"], "2x3: 18 a sqrt(c_2/6)")
    c("0,557465", rd["p_pentamero"], "diamantes: p = (2 + 3c)/4")
    c("0,686528", rd["H_p"], "diamantes: H(p)")
    c("0,07662", rd["p_pentamero"] * 4 / 3 - 2 / 3, "diamantes: c de la Proposición 29 de holonomia")
    c("0,184857", rd["a_gap_4"], "diamantes k = 4: a del testigo de mayor gap")
    c("27,487", 1 / rd["gap_norm_4"], "diamantes k = 4: 1/gap normalizado")
    pr = {float(r["eps"]): r for r in leer("regimen_degenerado.csv")}
    for eps, txt in ((1e-3, "4,67009"), (1e-4, "4,66963")):
        c(txt, float(pr[eps]["beta_sqrt_eps"]), f"2x3: beta sqrt(eps) en {eps:.0e}")
    c("5e-4", float(pr[1e-4]["beta_sqrt_eps"]) / rd["pred_beta_sqrt_eps"] - 1, "2x3: dif. rel. con la predicción")
    for eps, txt in ((1e-2, "0,5066"), (1e-3, "0,5720"), (1e-4, "0,5930")):
        c(txt, float(pr[eps]["x_peldano"]), f"2x3: multiplicador de peldaño en {eps:.0e}")
    c("3,152", float(pr[1e-2]["S_menos_Sast"]) / float(pr[1e-3]["S_menos_Sast"]), "2x3: caída de S - S* por década")
    c("3,163", float(pr[1e-3]["S_menos_Sast"]) / float(pr[1e-4]["S_menos_Sast"]), "2x3: caída de S - S* por década")
    fp = {float(r["t"]): r for r in leer("regimen_degenerado_fisher.csv")}
    c("0,28868", float(fp[0.01]["impar_sobre_t"]), "2x3: |X|_1 (parte impar)")
    c("0,82", float(fp[0.01]["par_sobre_t"]), "2x3: |Y|_1 (parte par)")
    c("0,6325", rd["s1_prisma"], "2x3: pendiente del pico s1")
    c("0,308618", rd["F_impar_prisma"], "2x3: Fisher SLD de la parte impar")
    c("0,30854", rd["sld_ext_prisma"], "2x3: SLD de la máxima entropía extrapolada")
    c("2e-4", abs(rd["sld_ext_prisma"] / rd["F_impar_prisma"] - 1), "2x3: dif. rel. SLD / Fisher de la parte impar")
    c("0,00121", rd["dS_cruce_001"], "2x3, eps = 1e-4: Delta S_G en t = 0,01")
    c("0,00127", rd["S2_1e-4"] * 0.01**2 / 2, "2x3, eps = 1e-4: 1/2 S'' t^2 en t = 0,01")
    c("0,09479", rd["dS_cruce_02"], "2x3, eps = 1e-4: Delta S_G en t = 0,2")
    c("0,10751", rd["Sast_02"], "2x3: S*(0,2) - S*(0)")
    c("0,5061", rd["S2_1e-4"] * 0.2**2 / 2, "2x3, eps = 1e-4: 1/2 S'' t^2 en t = 0,2")
    beta_1e4 = [r for r in leer("fisher_energia_libre_T0.csv") if int(r["L"]) == 3 and float(r["eps"]) == 1e-4][0]
    c("467,0", float(beta_1e4["beta"]), "2x3: beta en eps = 1e-4")
    di = {(int(r["k"]), r["metodo"]): r for r in leer("regimen_degenerado_diamantes.csv")}
    for k, txt in ((2, "0,3000"), (4, "0,4338"), (6, "0,4338")):
        c(txt, float(di[k, "teselas"]["gap"]), f"diamantes k = {k}: gap de H(0,185)")
    for k, txt in ((3, "1,5020"), (5, "1,9699"), (7, "1,9699")):
        c(txt, float(di[k, "fundamental"]["gap"]), f"diamantes k = {k}: gap del testigo óptimo")
    chk.check(all(abs(float(di[k, "programa lineal"]["S_ast"]) - float(di[k, "programa lineal"]["S_formula"])) < 1e-7
                  for k in (2, 3, 4)), "diamantes k = 2, 3, 4: S* del programa lineal = fórmula a 1e-7")
    ap = {(int(r["k"]), float(r["eps"])): r for r in leer("regimen_degenerado_aproximacion.csv")}
    for k, txts in ((4, ("29,60", "29,11", "28,79")), (3, ("21,58", "23,66", "25,08"))):
        for eps, txt in zip((1e-4, 1e-5, 1e-6), txts):
            c(txt, float(ap[k, eps]["beta_sobre_ln"]), f"diamantes k = {k}: beta/ln(1/eps) en {eps:.0e}")
    for eps, txt in ((1e-4, "0,185455"), (1e-5, "0,185338"), (1e-6, "0,185257")):
        c(txt, float(ap[4, eps]["a"]), f"diamantes k = 4: a de la máxima entropía en {eps:.0e}")

    # Cobertura
    sueltos = sorted(set(en_tex) - vistas - set(OTROS))
    chk.check(not sueltos, f"todo número del texto está comprobado o en OTROS; sin comprobar: {sueltos}")
    chk.finish()


if __name__ == "__main__":
    main()
