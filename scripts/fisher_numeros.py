"""Números del artículo de Fisher (solo lectura).

Artículo (~/fisica/holonomia_fisher.tex, «Holonomía y pegado de perspectivas de Werner II: información de
Fisher, primera ley del entrelazamiento y el testigo como hamiltoniano modular»): todas las secciones. No
resuelve ningún SDP ni diagonaliza nada: lee los CSV de fisher_kubo_mori.py.

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

    # Cobertura
    sueltos = sorted(set(en_tex) - vistas - set(OTROS))
    chk.check(not sueltos, f"todo número del texto está comprobado o en OTROS; sin comprobar: {sueltos}")
    chk.finish()


if __name__ == "__main__":
    main()
