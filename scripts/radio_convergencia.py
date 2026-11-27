"""Radio de convergencia de lambda_0(eps) (momento 0, S^z = 0, Theta = 0; sensibilidad_celda3.tex, problema 2).

Rayleigh-Schrödinger desde la suma de los tres patrones, eps = s z; test de la raíz |E_m|^{-1/m}.
Uso: python scripts/radio_convergencia.py n K s      (por ejemplo 6 120 0.1)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, scipy.sparse as sp
from celda_k_flujo import build, sector
n = int(sys.argv[1]); K = int(sys.argv[2]); s = float(sys.argv[3])  # s: reescala eps = s*z
N, diag, r, c = build(n, 0)
A = sp.coo_matrix((2*s*np.ones(r.size), (r, c)), shape=(N, N)).tocsr()
V = (A + A.T).tocsr()
codes, dig = sector(n, 0)
pat = np.isclose(diag, -n)
psi = [pat.astype(float)]; E = [-float(n)]
den = diag + n; den[pat] = 1.0
nrm = psi[0] @ psi[0]
for m in range(1, K+1):
    Em = psi[0] @ (V @ psi[m-1]) / nrm
    E.append(Em)
    rhs = -(V @ psi[m-1]) + sum(E[k]*psi[m-k] for k in range(1, m+1))
    x = rhs / den; x[pat] = 0.0
    psi.append(x)
E = np.array(E)
# coeficientes en eps: E_m / s^m
ks = np.arange(K+1)
for m in range(2, K+1, 2):
    if m % 8 == 0 or m == K:
        cm = abs(E[m]); cm2 = abs(E[m-2])
        print(f"m={m:3d}  |E_m|^(-1/m)*s = {s*cm**(-1/m):.4f}   sqrt(|E_m-2/E_m|)*s = {s*np.sqrt(cm2/cm) if cm>0 else np.nan:.4f}")
