#!/usr/bin/env python3
# Gerador de ressonancias da esfera core-shell fluida (nucleo A, casca B, meio M).
#
# Como a contagem funciona:
#
#   Toda curva comeca na malha base de 5000 pontos em x_M: 1000 pontos log em
#   [x_min, 0.1], 3000 lineares em [0.1, 0.9] e 1000 lineares em [0.9, 1]. A malha
#   passa um pouquinho de x_max so pra conseguir enxergar pico colado na borda.
#
#   alpha >= 1 e beta >= 1: aqui as ressonancias sao poucas e largas e a malha
#   da conta sozinha. N_res = maximos de W_T com proeminencia >= 0.1 em log10.
#
#   alpha < 1 ou beta < 1 (curva de risco): os picos ficam estreitos e se
#   amontoam, e a malha base subconta. Entao:
#     1. busca os zeros de D_l no plano complexo: Newton a partir dos minimos de
#        |D_l| no eixo real, conferido pelo principio do argumento (se nao
#        fechar, busca completa por subdivisao do retangulo). Eles ficam salvos
#        no dataset e servem pra localizar os modos;
#     2. em cada l compara os zeros com os maximos de W_l na malha base. Se nao
#        baterem, tira os zeros com Q < 2 e compara de novo; se ainda assim nao
#        bater (ou se nao teve busca de polos), fica com os maximos de W_l que
#        tem proeminencia >= 0.75;
#     3. refina a malha pelo passo que cada pico precisa. Com os polos, cada
#        ressonancia tem dx_nec = 2 sqrt(S/(k fundo) - gamma^2), k = 10^0.1 - 1:
#        o maior passo uniforme que garante ver o pico mesmo com o ponto mais
#        proximo a dx/2 do topo. Cada grupo de picos ganha um trecho uniforme com
#        o dx_nec do pico mais exigente (ou metade da separacao de Laurent, se
#        for menor). Os pontos nao sao colocados em cima dos picos. Se nao
#        couber em 2 milhoes de pontos, sobe um piso de passo ate caber;
#     4. maximos de W_T nessa malha (proeminencia >= 0.1) vao pra coluna
#        N_res_malha_refinada. O resumo tambem traz N_polos_visiveis (picos que
#        uma malha infinitamente fina veria), N_previsto_malha (quantos o passo
#        usado deveria ver), frac_malha_refinada, dx_necessario (passo pra ver
#        todos), dx_90 (passo pra ver 90%) e dx_piso_malha;
#     5. contagem final pelos polos, quando a busca fechou em todos os l. Perto
#        do polo z_k = x_k - i gamma_k, W_l(x) ~ S_k / ((x - x_k)^2 + gamma_k^2),
#        entao cada ressonancia tem posicao x_k, FWHM = 2 gamma_k,
#        Q = x_k/(2 gamma_k) e area pi S_k/gamma_k, tudo sem depender de malha.
#        O polo conta se Q >= 2 e se a area dele for >= 0.1% da energia integrada
#        numa janela de largura pi/Phi em volta dele (--frac-area muda isso).
#        N_res_polos_area conta modos (um por polo); N_res_polos_picos junta os
#        polos sobrepostos (vizinho a menos de gamma_1 + gamma_2), que na curva
#        aparecem como um pico so. Se a busca nao fechou, fica a malha refinada
#        do passo 4.
#        Polos com gamma/x < 1e-10 tem o gamma refeito so com valores no eixo
#        real (D_l = D_j + i D_y), porque o Newton em complexo devolve ruido.
#
# Exemplos:
#   python -u gerador_coreshell.py --regime velocity --workers 25
#   python -u gerador_coreshell.py --regime mixed --mix-factor 0.25 0.5 0.75 --workers 25
#   python -u gerador_coreshell.py --regime velocity --max-curvas 20     (teste rapido)

import argparse, json, os, signal, time, uuid, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import scipy.linalg as la
from scipy.optimize import minimize_scalar, newton
from scipy.signal import find_peaks, peak_widths
from scipy.special import spherical_jn, spherical_yn

warnings.filterwarnings("ignore")

LMAX = 10
X_MIN, X_MAX = 1e-4, 1.0
IMAG_MIN, IMAG_MAX = -0.5, 1e-3   # topo do retangulo acima do eixo: polo com gamma ~ 1e-30 nao fica colado na borda do contorno
DELTAS = np.array([0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.35, 1.50])
BINS_ETA = np.linspace(-1.0, 1.0, 21)
RHO_AGUA, C_AGUA = 1000.0, 1480.0

# malha e contagem
N_LOG, N_MEIO, N_FIM = 1000, 3000, 1000
PROM_WT = 0.1          # proeminencia minima (log10) dos picos de W_T
PROM_WL = 0.75         # proeminencia minima (log10) dos picos de W_l quando zeros e malha nao batem
Q_MIN = 2.0
FRAC_AREA = 1e-3       # curva de risco: polo conta se a area dele for >= 0.1% da energia integrada na janela de largura pi/Phi
                       # (calibrado em 30 curvas de risco: nenhum polo com Q >= 2 ficou abaixo de 0.3%, entao na pratica so o Q filtra)
Q_MIN_POLO = 2.0       # e se Q = x/(2 gamma) >= 2 (abaixo disso nao e' pico: a lorentziana nem vale e a malha nunca ve maximo)
DX_MIN = 1e-9          # abaixo disso o refino nao vale a pena (precisao do double)
N_MAX_PONTOS = 2_000_000

# busca de polos
NIVEIS_CONTORNO = (64, 128, 256)
NIVEIS_AUDITORIA = (256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
TOL_INTEIRO = 0.05
PROF_MAX = 40
BEYN_MAX, BEYN_FOLGA, BEYN_N = 8, 4, 500
RESIDUO_MAX = 1e-7
GAMMA_RAD = 1e-10      # gamma/x abaixo disso o gamma do Newton em complexo e' ruido; refaz pelo eixo real (polo_estreito)
TOL_DUPLICATA = 1e-8
LIMIAR_OVERFLOW_CASCA, LIMIAR_OVERFLOW_NUCLEO = 1e5, 1e5   # com o fundo do retangulo adaptado (imag_min) o contorno nao estoura mais; so pula em caso absurdo
VERSAO = 8


# ---------------------------------------------------------------------------
# Funcoes esfericas e coeficientes (ordem negativa via j_{-n-1} e y_{-n-1})
# ---------------------------------------------------------------------------
def J(n, x):
    if n >= 0: return spherical_jn(n, x)
    m = -n - 1; return (-1) ** (m + 1) * spherical_yn(m, x)

def Y(n, x):
    if n >= 0: return spherical_yn(n, x)
    m = -n - 1; return (-1) ** m * spherical_jn(m, x)

def dJ(n, x):
    if n >= 0: return spherical_jn(n, x, derivative=True)
    m = -n - 1; return (-1) ** (m + 1) * spherical_yn(m, x, derivative=True)

def dY(n, x):
    if n >= 0: return spherical_yn(n, x, derivative=True)
    m = -n - 1; return (-1) ** m * spherical_jn(m, x, derivative=True)

def H(n, x):
    if n >= 0: return spherical_jn(n, x) + 1j * spherical_yn(n, x)
    m = -n - 1; return 1j * (-1) ** m * (spherical_jn(m, x) + 1j * spherical_yn(m, x))

def dH(n, x):
    if n >= 0: return spherical_jn(n, x, derivative=True) + 1j * spherical_yn(n, x, derivative=True)
    m = -n - 1; return 1j * (-1) ** m * (spherical_jn(m, x, derivative=True) + 1j * spherical_yn(m, x, derivative=True))

# primitivas dos produtos de Bessel (energia da casca entre xB e yB, e do nucleo ate xA)
def Fjj(l, xB, yB): F = lambda x: x**3 * (J(l, x) ** 2 - J(l - 1, x) * J(l + 1, x)); return F(yB) - F(xB)
def Fyy(l, xB, yB): F = lambda x: x**3 * (Y(l, x) ** 2 - Y(l - 1, x) * Y(l + 1, x)); return F(yB) - F(xB)
def Fjy(l, xB, yB): F = lambda x: 0.5 * x**3 * (2 * J(l, x) * Y(l, x) - J(l + 1, x) * Y(l - 1, x) - J(l - 1, x) * Y(l + 1, x)); return F(yB) - F(xB)
def Bjj(l, xB, yB): B = lambda x: x**2 * J(l, x) * dJ(l, x); return B(yB) - B(xB)
def Byy(l, xB, yB): B = lambda x: x**2 * Y(l, x) * dY(l, x); return B(yB) - B(xB)
def Bjy(l, xB, yB): B = lambda x: x**2 * J(l, x) * dY(l, x); return B(yB) - B(xB)
def FA(l, xA): return 0.5 * xA**3 * (J(l, xA) ** 2 - J(l - 1, xA) * J(l + 1, xA))
def BA(l, xA): return xA**2 * J(l, xA) * dJ(l, xA)


def blocos(l, xA, xB, yM, yB, rho0, rho1, rho2, c0, c1, c2):
    rho_MB, rho_AB, k_MB, k_AB = rho0 / rho2, rho1 / rho2, c2 / c0, c2 / c1
    A, B, F = -J(l, yB), -Y(l, yB), rho_MB * H(l, yM)
    G, HH, JJ = -dJ(l, yB), -dY(l, yB), k_MB * dH(l, yM)
    L, M, P = -rho_AB * J(l, xA), J(l, xB), Y(l, xB)
    Q, R, S = -k_AB * dJ(l, xA), dJ(l, xB), dY(l, xB)
    D = (L * R - M * Q) * (F * HH - B * JJ) + (L * S - P * Q) * (A * JJ - F * G)
    return D, L, M, P, Q, R, S, rho_MB, k_MB


def coeffs(l, xA, xB, yM, yB, *mat):
    D, L, M, P, Q, R, S, rho_MB, k_MB = blocos(l, xA, xB, yM, yB, *mat)
    X = 1j * rho_MB * k_MB / yM**2
    return (M * S - P * R) * X / D, (P * Q - L * S) * X / D, (L * R - M * Q) * X / D


def argumentos(z, delta, mat):
    rho0, rho1, rho2, c0, c1, c2 = mat
    xB = (c0 / c2) * z
    return (c0 / c1) * z, xB, (1.0 + delta) * z, (1.0 + delta) * xB


NORMALIZACAO = "ZM"   # "ZM": W_T = alpha W~_A + beta W~_B (tudo normalizado por Z_M); "separada": W~_A + W~_B (versoes antigas)

def pesos_Z(mat):
    """Pesos do nucleo e da casca na energia total. Por definicao, W~_A = 2 f0 W_A/(Z_A |A|^2) e W~_B = 2 f0 W_B/(Z_B |A|^2).
    Pra somar as duas com a mesma normalizacao (Z_M):
        2 f0 (W_A + W_B)/(Z_M |A|^2) = (Z_A/Z_M) W~_A + (Z_B/Z_M) W~_B = alpha W~_A + beta W~_B.
    Com alpha = beta os dois jeitos so diferem por um fator comum (os picos nao mudam)."""
    if NORMALIZACAO == "separada": return 1.0, 1.0
    rho0, rho1, rho2, c0, c1, c2 = mat; ZM = rho0 * c0
    return rho1 * c1 / ZM, rho2 * c2 / ZM


def energia_l(l, x, delta, mat):
    """W_l = WK + WP de um l so (nucleo + casca), com os pesos de pesos_Z."""
    x = np.asarray(x, dtype=float); xA, xB, yM, yB = argumentos(x, delta, mat); f = 2 * l + 1
    with np.errstate(all="ignore"):
        bl, cl, dl = coeffs(l, xA, xB, yM, yB, *mat)
        nucleo = np.real(np.abs(bl) ** 2 * f * (2.0 * FA(l, xA) + BA(l, xA)))
        cl2, dl2, cd = np.abs(cl) ** 2, np.abs(dl) ** 2, np.real(np.conj(cl) * dl)
        pot = 0.5 * cl2 * Fjj(l, xB, yB) + 0.5 * dl2 * Fyy(l, xB, yB) + cd * Fjy(l, xB, yB)
        borda = cl2 * Bjj(l, xB, yB) + dl2 * Byy(l, xB, yB) + 2.0 * cd * Bjy(l, xB, yB)
        wA, wB = pesos_Z(mat)
        w = wA * nucleo + wB * np.real(f * (2.0 * pot + borda))
    return np.where(np.isfinite(w), w, np.nan)


def energia_total(x, delta, mat, lmax=LMAX, bloco=400_000):
    """W_T = soma de W_l, l = 0..lmax. Mesma conta do energia_l, mas cada Bessel e' calculada uma vez so (ordens -1 a
    lmax+1 em cada argumento) e as derivadas saem da recorrencia f_l' = f_{l-1} - (l+1)/x f_l. Conferido contra a soma
    de energia_l: diferenca relativa <= 5e-11, uns 6x mais rapido."""
    x = np.asarray(x, dtype=float)
    if len(x) > bloco: return np.concatenate([energia_total(x[i:i + bloco], delta, mat, lmax) for i in range(0, len(x), bloco)])
    rho0, rho1, rho2, c0, c1, c2 = mat; xA, xB, yM, yB = argumentos(x, delta, mat); wA, wB = pesos_Z(mat)
    rho_MB, rho_AB, k_MB, k_AB = rho0 / rho2, rho1 / rho2, c2 / c0, c2 / c1
    ordens = range(-1, lmax + 2)
    with np.errstate(all="ignore"):
        JA = {n: J(n, xA) for n in ordens}; JB = {n: J(n, xB) for n in ordens}; YB = {n: Y(n, xB) for n in ordens}
        Jb = {n: J(n, yB) for n in ordens}; Yb = {n: Y(n, yB) for n in ordens}; JM = {n: J(n, yM) for n in range(-1, lmax + 1)}; YM = {n: Y(n, yM) for n in range(-1, lmax + 1)}
        W = np.zeros_like(x)
        for l in range(lmax + 1):
            d = lambda F, z: F[l - 1] - (l + 1) / z * F[l]
            jA, djA, jB, djB, yB_, dyB = JA[l], d(JA, xA), JB[l], d(JB, xB), YB[l], d(YB, xB)
            jb, djb, yb, dyb = Jb[l], d(Jb, yB), Yb[l], d(Yb, yB)
            h, dh = JM[l] + 1j * YM[l], d(JM, yM) + 1j * d(YM, yM)
            A, B, F, G, HH, JJ = -jb, -yb, rho_MB * h, -djb, -dyb, k_MB * dh
            L, M, P, Q, R, S = -rho_AB * jA, jB, yB_, -k_AB * djA, djB, dyB
            D = (L * R - M * Q) * (F * HH - B * JJ) + (L * S - P * Q) * (A * JJ - F * G); X = 1j * rho_MB * k_MB / yM**2 / D
            bl, cl, dl = (M * S - P * R) * X, (P * Q - L * S) * X, (L * R - M * Q) * X; f = 2 * l + 1
            FA = 0.5 * xA**3 * (jA**2 - JA[l - 1] * JA[l + 1]); BA = xA**2 * jA * djA
            Fp = lambda Jf, Yf, z, jl, yl: (z**3 * (jl**2 - Jf[l - 1] * Jf[l + 1]), z**3 * (yl**2 - Yf[l - 1] * Yf[l + 1]),
                                            0.5 * z**3 * (2 * jl * yl - Jf[l + 1] * Yf[l - 1] - Jf[l - 1] * Yf[l + 1]))
            Fjj_b, Fyy_b, Fjy_b = Fp(Jb, Yb, yB, jb, yb); Fjj_a, Fyy_a, Fjy_a = Fp(JB, YB, xB, jB, yB_)
            Fjj, Fyy, Fjy = Fjj_b - Fjj_a, Fyy_b - Fyy_a, Fjy_b - Fjy_a
            Bjj = yB**2 * jb * djb - xB**2 * jB * djB; Byy = yB**2 * yb * dyb - xB**2 * yB_ * dyB; Bjy = yB**2 * jb * dyb - xB**2 * jB * dyB
            cl2, dl2, cd = np.abs(cl) ** 2, np.abs(dl) ** 2, np.real(np.conj(cl) * dl)
            w = wA * np.real(np.abs(bl) ** 2 * f * (2 * FA + BA)) + wB * np.real(f * (cl2 * Fjj + dl2 * Fyy + 2 * cd * Fjy + cl2 * Bjj + dl2 * Byy + 2 * cd * Bjy))
            W += np.nan_to_num(np.where(np.isfinite(w), w, np.nan))
    return np.where(np.isfinite(W) & (W > 0), W, 0.0)


def D_l(l, z, delta, mat):
    """z^4 D_l: tem os mesmos zeros, mas sem o polo de ordem 4 em z = 0 colado em x_min."""
    z = np.asarray(z, dtype=complex)
    with np.errstate(all="ignore"):
        return z**4 * blocos(l, *argumentos(z, delta, mat), *mat)[0]


def D_partes(l, x, delta, mat):
    """No eixo real so F e JJ sao complexos (vem de h_l = j_l + i y_l, fora da esfera) e D_l e' linear em (F, JJ).
    Entao D_l(x) = D_j(x) + i D_y(x), com D_j = D_l trocando h_l por j_l e D_y = D_l trocando h_l por y_l.
    As duas partes sao reais e calculadas separadas, cada uma com a precisao do double."""
    x = np.asarray(x, dtype=float); xA, xB, yM, yB = argumentos(x, delta, mat); rho0, rho1, rho2, c0, c1, c2 = mat
    rho_MB, rho_AB, k_MB, k_AB = rho0 / rho2, rho1 / rho2, c2 / c0, c2 / c1
    A, B, G, HH = -J(l, yB), -Y(l, yB), -dJ(l, yB), -dY(l, yB)
    C1 = -rho_AB * J(l, xA) * dJ(l, xB) + J(l, xB) * k_AB * dJ(l, xA)          # L R - M Q
    C2 = -rho_AB * J(l, xA) * dY(l, xB) + Y(l, xB) * k_AB * dJ(l, xA)          # L S - P Q
    D = lambda F, JJ: C1 * (F * HH - B * JJ) + C2 * (A * JJ - F * G)
    return D(rho_MB * J(l, yM), k_MB * dJ(l, yM)), D(rho_MB * Y(l, yM), k_MB * dY(l, yM))


def polo_estreito(l, z, delta, mat):
    """Polo colado no eixo real (gamma/x < GAMMA_RAD): o Newton em complexo devolve gamma ~ 1e-20 de ruido, porque
    Im de j_l(x + i eps) se perde. Aqui o Newton usa so valores no eixo real:
        D_l(x0 - i gamma) ~ D_l(x0) - i gamma D_l'(x0) = 0  ->  z0 = x0 - D_l(x0)/D_l'(x0),  D_l = D_j + i D_y,
    itera x0 ate Re(D/D') = 0 e fica gamma = Im(D_l(x0)/D_l'(x0)). Conferido com mpmath a 60 digitos."""
    x = z.real
    if -z.imag > GAMMA_RAD * x: return z
    Dc = lambda t: (lambda p: p[0] + 1j * p[1])(D_partes(l, t, delta, mat))
    for _ in range(50):
        h = 1e-7 * x; q = complex(Dc(x) / ((Dc(x + h) - Dc(x - h)) / (2 * h))); x -= q.real
        if abs(q.real) < 1e-15 * x: break
    return complex(x, -q.imag) if np.isfinite(q) and q.imag > 0 else z


def peso_polo(l, z, delta, mat):
    """Perto do polo z = x_k - i gamma: b_l ~ R_b/(x - z), c_l ~ R_c/(x - z), d_l ~ R_d/(x - z), com R = N(z)/D_l'(z).
    Entao W_l(x) ~ S / ((x - x_k)^2 + gamma^2), com
        S = (2l+1) [ |R_b|^2 (2 FA + BA) + |R_c|^2 (Fjj + Bjj) + |R_d|^2 (Fyy + Byy) + 2 Re(R_c* R_d)(Fjy + Bjy) ]  em x_k.
    Altura do pico = S/gamma^2, largura a meia altura = 2 gamma e area = pi S/gamma. Nada disso depende de malha."""
    z = complex(z); h = 1e-7 * abs(z)
    with np.errstate(all="ignore"):
        _, L, M, P, Q, R, S_, rho_MB, k_MB = blocos(l, *argumentos(z, delta, mat), *mat)
        dD = (blocos(l, *argumentos(z + h, delta, mat), *mat)[0] - blocos(l, *argumentos(z - h, delta, mat), *mat)[0]) / (2 * h)
        X = 1j * rho_MB * k_MB / ((1 + delta) * z) ** 2 / dD
        Rb, Rc, Rd = (M * S_ - P * R) * X, (P * Q - L * S_) * X, (L * R - M * Q) * X
        xA, xB, yM, yB = argumentos(z.real, delta, mat)
        wA, wB = pesos_Z(mat)
        S = (2 * l + 1) * (wA * abs(Rb) ** 2 * (2 * FA(l, xA) + BA(l, xA)) + wB * (abs(Rc) ** 2 * (Fjj(l, xB, yB) + Bjj(l, xB, yB))
                           + abs(Rd) ** 2 * (Fyy(l, xB, yB) + Byy(l, xB, yB)) + 2 * np.real(np.conj(Rc) * Rd) * (Fjy(l, xB, yB) + Bjy(l, xB, yB))))
    return float(np.real(S))


# ---------------------------------------------------------------------------
# Materiais e grade
# ---------------------------------------------------------------------------
def materiais(alpha, beta, regime, f=0.5):
    ZM = RHO_AGUA * C_AGUA
    if regime == "velocity":
        r = RHO_AGUA; return r, r, r, ZM / r, alpha * ZM / r, beta * ZM / r
    if regime == "density":
        c = C_AGUA; return ZM / c, alpha * ZM / c, beta * ZM / c, c, c, c
    return RHO_AGUA, RHO_AGUA * alpha**f, RHO_AGUA * beta**f, ZM / RHO_AGUA, C_AGUA * alpha ** (1 - f), C_AGUA * beta ** (1 - f)


def grade_eta(por_bin=20, seed=42):
    vals = np.unique(np.concatenate([np.linspace(a, b, 10) for a, b in ((0.001, 0.009), (0.01, 0.09), (0.1, 0.9), (1, 9), (10, 90), (100, 900))] + [[1000.0, 2000.0, 3000.0]]))
    A, B = (m.ravel() for m in np.meshgrid(vals, vals, indexing="ij")); eta = (A - B) / (A + B)
    b = np.clip(np.searchsorted(BINS_ETA, eta, side="left") - 1, 0, len(BINS_ETA) - 2); b[eta == BINS_ETA[0]] = 0
    rng = np.random.default_rng(seed); alphas, betas = [], []
    for i in range(len(BINS_ETA) - 1):
        aa, bb = A[b == i], B[b == i]
        if aa.size > por_bin:
            k = np.linspace(0, aa.size - 1, por_bin, dtype=int); aa, bb = aa[k], bb[k]
        elif aa.size < por_bin:
            meio = 0.5 * (BINS_ETA[i] + BINS_ETA[i + 1]); s = rng.choice(vals, size=por_bin - aa.size, replace=True)
            aa, bb = np.concatenate([aa, s * (1 + meio) / 2]), np.concatenate([bb, s * (1 - meio) / 2])
        alphas.append(aa); betas.append(bb)
    return np.concatenate(alphas), np.concatenate(betas)


# ---------------------------------------------------------------------------
# Malha e picos
# ---------------------------------------------------------------------------
def malha_base(delta, mat, x_min=X_MIN, x_max=X_MAX):
    rho0, rho1, rho2, c0, c1, c2 = mat
    esp = np.pi / (c0 / c1 + delta * c0 / c2)        # distancia media entre modos de um mesmo l (pi/Phi)
    ext = min(esp, 0.05 * x_max)
    a, b = min(0.1, x_max), min(0.9, x_max)
    x = np.unique(np.concatenate([np.geomspace(x_min, a, N_LOG), np.linspace(a, b, N_MEIO), np.linspace(b, x_max + ext, N_FIM)]))
    return x, esp


def maximos(energia, x, esp, prom=None, x_min=X_MIN, x_max=X_MAX, w=None):
    """Indices e posicoes dos maximos de energia(x). Pico perto de x_max tem a posicao refinada antes de
    decidir se fica dentro do intervalo."""
    w = energia(x) if w is None else w
    if prom is None:
        idx = find_peaks(np.where(np.isfinite(w), w, -np.inf))[0]
    else:
        ok = np.isfinite(w) & (w > 0)
        if not ok.any(): return np.array([], int), np.array([])
        piso = max(w[ok].min() * 1e-14, np.finfo(float).tiny)
        idx = find_peaks(np.log10(np.where(ok, np.maximum(w, piso), piso)), prominence=prom)[0]
    fica, xs = [], []
    for i in idx:
        xp = float(x[i])
        if xp > x_max - 2 * min(esp, 0.05 * x_max) and 0 < i < len(x) - 1:
            alvo = lambda t: -np.log(max(float(energia(np.array([t]))[0]), np.finfo(float).tiny))
            xp = float(minimize_scalar(alvo, bounds=(x[i - 1], x[i + 1]), method="bounded", options={"xatol": 1e-12}).x)
        if x_min <= xp <= x_max: fica.append(i); xs.append(xp)
    return np.array(fica, int), np.array(xs)


def refina_posicoes(energia, x, xs, it=60):
    """Secao aurea em todos os picos de uma vez, cada um no seu intervalo [x_{i-1}, x_{i+1}]."""
    if len(xs) == 0: return xs
    i = np.clip(np.searchsorted(x, xs), 1, len(x) - 1)
    a, b = x[i - 1], x[np.minimum(i + 1, len(x) - 1)]; g = (np.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a); fc, fd = energia(c), energia(d)
    for _ in range(it):
        m = np.nan_to_num(fc, nan=-np.inf) > np.nan_to_num(fd, nan=-np.inf)
        b, a = np.where(m, d, b), np.where(m, a, c)
        c, d = b - g * (b - a), a + g * (b - a); fc, fd = energia(c), energia(d)
    return 0.5 * (a + b)


def medir_picos(WT, x, xs, esp, escala=None, n_scan=80, it=50):
    """Mede cada pico contado do mesmo jeito que o gerador antigo media na energia: topo refinado, fundo nas
    bordas da janela do pico (metade do caminho ate os picos vizinhos), e largura a meia altura procurando o
    cruzamento de cada lado. Tudo vetorizado, todos os picos de uma vez."""
    n = len(xs)
    if n == 0: return []
    xs0 = np.asarray(xs, dtype=float); w0 = WT(xs0); xs = refina_posicoes(WT, x, xs0); wp = WT(xs)
    xs, wp = np.where(wp >= w0, xs, xs0), np.maximum(wp, w0)     # se o refino escorregar pra fora de um pico muito fino, fica o ponto da malha
    viz_e = np.r_[X_MIN, 0.5 * (xs[1:] + xs[:-1])]; viz_d = np.r_[0.5 * (xs[1:] + xs[:-1]), X_MAX + min(esp, 0.05)]
    lo, hi = np.maximum(viz_e, xs - 0.5 * esp), np.minimum(viz_d, xs + 0.5 * esp)
    # fundo: mediana de W_T em pontos perto das duas bordas da janela
    fr = np.array([0.02, 0.06, 0.12, 0.2])
    pts = np.concatenate([lo[:, None] + fr * (xs - lo)[:, None], hi[:, None] - fr * (hi - xs)[:, None]], axis=1)
    fundo = np.median(WT(pts.ravel()).reshape(n, -1), axis=1)
    nivel = fundo + 0.5 * (wp - fundo)
    escala = np.full(n, np.nan) if escala is None else np.asarray(escala, dtype=float)
    lados = []
    for lado, dmax in ((-1, xs - lo), (+1, hi - xs)):
        base = np.where(np.isfinite(escala) & (escala > 0), escala, dmax / 100)
        dmin = np.maximum(32 * np.finfo(float).eps * np.maximum(1, xs), np.minimum(1e-3 * base, 1e-3 * dmax))
        D = np.exp(np.linspace(np.log(dmin), np.log(np.maximum(dmax, 2 * dmin)), n_scan, axis=1))
        abaixo = (WT((xs[:, None] + lado * D).ravel()).reshape(n, -1) < nivel[:, None])
        tem = abaixo.any(axis=1); k = np.argmax(abaixo, axis=1)
        a = np.where(k > 0, D[np.arange(n), np.maximum(k - 1, 0)], 0.0); b = D[np.arange(n), k]
        for _ in range(it):                     # bissecao entre o ultimo ponto acima e o primeiro abaixo do nivel
            m = 0.5 * (a + b); acima = WT(xs + lado * m) >= nivel; a, b = np.where(acima, m, a), np.where(acima, b, m)
        lados.append(np.where(tem, 0.5 * (a + b), np.nan))
    fwhm = lados[0] + lados[1]; ok = (wp > fundo) & (fundo > 0)
    saida = []
    for i in range(n):
        razao = wp[i] / fundo[i] if fundo[i] > 0 else np.inf
        saida.append({"xM_peak": float(xs[i]), "WT_peak": float(wp[i]), "WT_background": float(fundo[i]), "energy_ratio": float(razao),
                      "energy_log10_prominence": float(np.log10(razao)) if razao > 0 else np.nan,
                      "FWHM_energy": float(fwhm[i]) if ok[i] and np.isfinite(fwhm[i]) else np.nan,
                      "Q_energy": float(xs[i] / fwhm[i]) if ok[i] and np.isfinite(fwhm[i]) and fwhm[i] > 0 else np.nan})
    return saida


def casar(xs, zeros, esp):
    """Casa maximos da malha com zeros, 1 a 1, do par mais proximo pro mais distante."""
    pares = sorted((abs(x - z.real), i, j) for i, x in enumerate(xs) for j, z in enumerate(zeros) if abs(x - z.real) <= max(2 * abs(z.imag), 0.25 * esp))
    usado_x, usado_z, casados = set(), set(), {}
    for _, i, j in pares:
        if i not in usado_x and j not in usado_z: usado_x.add(i); usado_z.add(j); casados[j] = i
    return casados


def modos_por_l(delta, mat, zeros_por_l, x, esp, lmax=LMAX):
    """Posicoes dos modos de cada l, decididas pela comparacao zeros x maximos de W_l."""
    saida = []
    for l in range(lmax + 1):
        zeros = zeros_por_l.get(l); E = lambda v, l=l: energia_l(l, v, delta, mat)
        _, xs = maximos(E, x, esp)
        r = {"l": l, "N_zeros": None if zeros is None else len(zeros), "N_max_malha": len(xs)}
        if zeros is not None:
            cas = casar(xs, zeros, esp)
            if len(xs) == len(zeros) == len(cas):
                r.update(regra="bate", x=[z.real for z in zeros]); saida.append(r); continue
            bons = [j for j, z in enumerate(zeros) if z.real / (2 * abs(z.imag)) >= Q_MIN]
            sobra_x = [v for i, v in enumerate(xs) if i not in {cas[j] for j in cas if j not in bons}]
            z2 = [zeros[j] for j in bons]
            if len(sobra_x) == len(z2) == len(casar(sobra_x, z2, esp)):
                r.update(regra="bate_Q2", x=[z.real for z in z2]); saida.append(r); continue
        _, xp = maximos(E, x, esp, prom=PROM_WL)
        r.update(regra="proeminencia" if zeros is not None else "sem_polos", x=list(refina_posicoes(E, x, xp)))
        saida.append(r)
    return saida


def separacao_laurent(polos, delta, mat, n=400):
    """Proposta do Gilberto: perto de dois polos vizinhos, 1/D_l ~ A1/(x-z1) + A2/(x-z2), com A = 1/D_l'(z).
    Procura no eixo real onde esse modelo de dois polos tem os seus maximos e devolve a distancia entre eles
    (nan quando so sobra um maximo, ou seja, os dois picos se fundem e nenhuma malha separa).
    Polos do mesmo l somam na amplitude; de l diferentes somam na energia (duas lorentzianas)."""
    if len(polos) < 2: return np.array([]), np.array([])
    pol = sorted(polos, key=lambda p: p["pole_real"])
    z = np.array([complex(p["pole_real"], p["pole_imag"]) for p in pol]); l = np.array([p["l"] for p in pol])
    A, h = np.empty(len(z), complex), np.empty(len(z))
    for ll in np.unique(l):
        k = l == ll; zz = z[k]; e = 1e-7 * np.maximum(np.abs(zz), 1e-4)
        A[k] = 2 * e / (D_l(ll, zz + e, delta, mat) - D_l(ll, zz - e, delta, mat))
        gam = np.abs(zz.imag); s_ = np.maximum(gam, 1e-6 * zz.real)          # altura do pico tirada da cauda lorentziana
        h[k] = energia_l(ll, zz.real + s_, delta, mat) * (s_**2 + gam**2) / gam**2
    z1, z2 = z[:-1], z[1:]; x1, x2 = z1.real, z2.real
    G1, G2 = np.abs(z1.imag)[:, None], np.abs(z2.imag)[:, None]
    g1, g2 = np.maximum(G1[:, 0], 1e3 * np.finfo(float).eps * x1), np.maximum(G2[:, 0], 1e3 * np.finfo(float).eps * x2)
    u = np.geomspace(1e-3, 30, n // 4)
    X = np.sort(np.concatenate([x1[:, None] - g1[:, None] * u[::-1], x1[:, None] + g1[:, None] * u, np.linspace(x1, x2, n // 4 + 2, axis=1)[:, 1:-1],
                                x2[:, None] - g2[:, None] * u[::-1], x2[:, None] + g2[:, None] * u], axis=1), axis=1)
    mesmo = (l[:-1] == l[1:])[:, None]
    with np.errstate(all="ignore"):
        W = np.where(mesmo, np.abs(A[:-1, None] / (X - z1[:, None]) + A[1:, None] / (X - z2[:, None])) ** 2,
                     h[:-1, None] * G1**2 / ((X - x1[:, None]) ** 2 + G1**2) + h[1:, None] * G2**2 / ((X - x2[:, None]) ** 2 + G2**2))
    mx = (W[:, 1:-1] > W[:, :-2]) & (W[:, 1:-1] >= W[:, 2:]) & np.isfinite(W[:, 1:-1])
    sep = np.full(len(z1), np.nan)
    for i in np.flatnonzero(mx.sum(axis=1) >= 2):
        p = X[i, 1:-1][mx[i]]; sep[i] = p.max() - p.min()
    return 0.5 * (x1 + x2), sep


def malha_refinada(x, posicoes, esp, laurent=None):
    """Malha base + malha local em cada grupo de modos proximos + um ponto em cima de cada modo.
    Com laurent = (meio_do_par, separacao), o dx do grupo vem da separacao de Laurent dos pares que caem nele
    (pares fundidos nao entram, nao adianta refinar pra eles); sem isso, metade da menor distancia entre modos."""
    P = np.sort(np.asarray(posicoes, dtype=float))
    if len(P) < 2: return x, True, np.nan
    extras, coube, dx_min = [], True, np.inf
    for G in np.split(P, np.flatnonzero(np.diff(P) > 0.25 * esp) + 1):
        if len(G) < 2: continue
        dx = 0.5 * np.min(np.diff(G))
        if laurent is not None and len(laurent[0]):
            k = (laurent[0] >= G[0]) & (laurent[0] <= G[-1]) & np.isfinite(laurent[1])
            if k.any(): dx = 0.5 * np.min(laurent[1][k])
        if dx < DX_MIN: coube, dx = False, DX_MIN
        dx_min = min(dx_min, dx); lo, hi = G[0] - 10 * dx, G[-1] + 10 * dx
        extras.append(np.linspace(lo, hi, int(np.ceil((hi - lo) / dx)) + 1))
    viz = np.minimum(np.r_[np.inf, np.diff(P)], np.r_[np.diff(P), np.inf]); h = 0.5 * np.minimum(viz, 0.25 * esp)
    extras.append(np.concatenate([P - h, P, P + h]))
    n = sum(map(len, extras))
    if n + len(x) > N_MAX_PONTOS:      # nao cabe: afina o que der dentro do limite
        coube, k = False, (N_MAX_PONTOS - len(x)) / n
        extras = [np.linspace(e[0], e[-1], max(3, int(len(e) * k))) for e in extras]
    return np.unique(np.concatenate([x] + extras)), coube, float(dx_min)


# ---------------------------------------------------------------------------
# Busca de polos (principio do argumento + Beyn + Newton)
# ---------------------------------------------------------------------------
def contorno(xa, xb, ya, yb, n):
    t = np.linspace(0, 1, max(int(n), 4), endpoint=False)
    return np.concatenate([xa + (xb - xa) * t + 1j * ya, xb + 1j * (ya + (yb - ya) * t), xb - (xb - xa) * t + 1j * yb, xa + 1j * (yb - (yb - ya) * t)])


def voltas(F, caixa, n):
    v = np.asarray(F(contorno(*caixa, n)), dtype=complex)
    if not np.all(np.isfinite(v)) or np.any(v == 0): return None
    with np.errstate(all="ignore"): r = np.roll(v, -1) / v
    return float(np.sum(np.angle(r)) / (2 * np.pi)) if np.all(np.isfinite(r)) else None


def conta_zeros(F, caixa, niveis=NIVEIS_CONTORNO):
    """Numero de zeros na caixa; so aceita quando duas resolucoes seguidas dao o mesmo inteiro."""
    antes = None
    for n in niveis:
        w = voltas(F, caixa, n)
        k = None if w is None else int(round(w))
        if k is None or k < 0 or abs(w - k) > TOL_INTEIRO: antes = None; continue
        if k == antes: return k
        antes = k
    return None


def auditoria(F, caixa, achados):
    """Recontagem do contorno inteiro com resolucao crescente; passa se bater duas vezes com o que foi achado."""
    seguidas, ultimo = 0, None
    for n in NIVEIS_AUDITORIA:
        w = voltas(F, caixa, n)
        k = None if w is None else int(round(w))
        if k is None or k < 0 or abs(w - k) > TOL_INTEIRO: seguidas = 0; continue
        ultimo = k; seguidas = seguidas + 1 if k == achados else 0
        if seguidas >= 2: return k, True
    return ultimo, False


def dentro(z, xa, xb, ya, yb, tol=1e-10): return xa - tol <= z.real <= xb + tol and ya - tol <= z.imag <= yb + tol


def residuo(F, z):
    v = F(z); h = 1e-7 * max(1.0, abs(z))
    viz = np.asarray(F(np.array([z + h, z - h, z + 1j * h, z - 1j * h])), dtype=complex); viz = viz[np.isfinite(viz)]
    return abs(v) / max(np.median(np.abs(viz)), np.finfo(float).tiny) if np.isfinite(v) and len(viz) else np.inf


def polir(F, z0, caixa):
    try: z = complex(newton(F, z0, tol=1e-13, maxiter=150))
    except Exception: return None
    return z if dentro(z, *caixa) and residuo(F, z) <= RESIDUO_MAX else None


def beyn(F, caixa, r):
    """Momentos da integral de contorno -> autovalores = zeros. So usado pra caixas com 2 a 8 zeros."""
    L = r + BEYN_FOLGA; xa, xb, ya, yb = caixa; c = 0.5 * (xa + xb) + 0.5j * (ya + yb)
    z = contorno(*caixa, BEYN_N); fz = np.asarray(F(z), dtype=complex)
    if not np.all(np.isfinite(fz)) or np.any(fz == 0): return None
    dlog = (F(z + 1e-6) - F(z - 1e-6)) / 2e-6 / fz; dz = np.roll(z, -1) - z; w = z - c
    m = [np.sum(0.5 * ((w**k) * dlog + np.roll((w**k) * dlog, -1)) * dz) / (2j * np.pi) for k in range(2 * L)]
    H0 = np.array([[m[i + j] for j in range(L)] for i in range(L)]); H1 = np.array([[m[i + j + 1] for j in range(L)] for i in range(L)])
    try:
        U, S, Vh = la.svd(H0)
        if np.any(S[:r] <= 0): return None
        return la.eigvals(U[:, :r].conj().T @ H1 @ Vh[:r].conj().T @ np.diag(1 / S[:r])) + c
    except Exception:
        return None


def sem_duplicatas(zs):
    out = []
    for z in sorted(zs, key=lambda z: (z.real, z.imag)):
        if not out or min(abs(z - u) for u in out) > TOL_DUPLICATA: out.append(z)
    return out


def zeros_na_caixa(F, caixa, prazo, prof=0):
    if time.time() >= prazo: return []
    k = conta_zeros(F, caixa); xa, xb, ya, yb = caixa
    if k == 0: return []
    if k == 1:
        z = polir(F, 0.5 * (xa + xb) + 0.5j * (ya + yb), caixa)
        if z is not None: return [z]
    elif k is not None and k <= BEYN_MAX:
        cand = beyn(F, caixa, k)
        if cand is not None and len(cand) == k and all(dentro(c, *caixa) for c in cand):
            pol = [polir(F, c, caixa) for c in cand]
            if all(p is not None for p in pol) and len(sem_duplicatas(pol)) == k: return sem_duplicatas(pol)
    if prof >= PROF_MAX: return []
    xm, ym = 0.5 * (xa + xb), 0.5 * (ya + yb)
    achados = []
    for sub in ((xa, xm, ya, ym), (xm, xb, ya, ym), (xa, xm, ym, yb), (xm, xb, ym, yb)):
        achados += zeros_na_caixa(F, sub, prazo, prof + 1)
    return sem_duplicatas(achados)


def risco_overflow(alpha, beta, delta, regime):
    """Quando o som no nucleo/casca e' muito mais lento que na agua, os argumentos das Bessel ficam enormes no
    plano complexo e o D_l perde precisao. Nesses casos a busca de polos e' pulada."""
    if regime == "velocity": casca, nucleo = (1 + delta) / beta, 1 / alpha
    elif regime == "mixed": casca, nucleo = (1 + delta) / np.sqrt(beta), 1 / np.sqrt(alpha)
    else: return False
    return casca > LIMIAR_OVERFLOW_CASCA or nucleo > LIMIAR_OVERFLOW_NUCLEO


def no_eixo(z):
    """Zero com 0 <= Im z < GAMMA_RAD |z|: nao existe polo no semiplano de cima (sistema passivo), entao e' um polo
    colado no eixo cujo gamma o Newton nao resolve (o sinal de Im e' ruido). Fica com Im negativo minusculo e o
    polo_estreito refaz o gamma."""
    return complex(z.real, -1e-300) if 0 <= z.imag < GAMMA_RAD * abs(z) else z


def zeros_rapido(F, esp, caixa, prazo):
    """Primeira tentativa, barata: sementes nos minimos de |z^4 D_l| numa malha fina do eixo real (passo ~ esp/200,
    bem menor que a distancia entre zeros de um mesmo l) e Newton a partir de cada uma."""
    xr = np.linspace(X_MIN, X_MAX, int(min(2e6, max(2e5, 200 / esp))))
    a = np.abs(F(xr.astype(complex))); sem = xr[np.flatnonzero((a[1:-1] < a[:-2]) & (a[1:-1] <= a[2:])) + 1]; zs = []
    for s0 in sem:
        if time.time() >= prazo: return None
        for y0 in (-1e-12, -1e-9, -1e-6, -1e-3):
            z = polir(F, complex(s0, y0), caixa)
            if z is not None: zs.append(z); break
    return zs


def buscar_polos(delta, mat, prazo, x, esp, lmax=LMAX):
    """Zeros de z^4 D_l para cada l no retangulo [x_min, x_max] x [imag_min, imag_max].
    1. Newton a partir dos minimos de |D_l| no eixo real (rapido, acha quase tudo);
    2. auditoria pelo principio do argumento no retangulo inteiro;
    3. se a contagem nao fechar, busca completa: retangulo cortado em faixas com borda nos maximos de |D_l| no eixo,
       cada faixa subdividida ate ter 1 zero (Newton) ou ate 8 (Beyn). Junta com o que o passo 1 achou e audita de novo.
    No fim, todo polo com gamma/x < 1e-10 tem o gamma refeito pelo eixo real (polo_estreito)."""
    # fundo do retangulo: com som lento no nucleo/casca, Im dos argumentos das Bessel cresce como k * Im(z) e o
    # produto de Bessel em D_l estoura ou zera no double. Comeca com |Im(argumento)| <= 150 e sobe se precisar.
    rho0, rho1, rho2, c0, c1, c2 = mat; y0 = -min(-IMAG_MIN, 150.0 / max(c0 / c1, (1 + delta) * c0 / c2, 1 + delta))
    polos, zeros_por_l, info = [], {}, {"imag_min": {}}
    xr = x[x <= X_MAX]
    for l in range(lmax + 1):
        if time.time() >= prazo: zeros_por_l[l] = None; info[l] = "sem_tempo"; continue
        F = lambda z, l=l: D_l(l, z, delta, mat)
        ymin = y0                                    # se D_l zera ou estoura no fundo do contorno, sobe o fundo pela metade
        while ymin < -1e-4:
            v = F(contorno(X_MIN, X_MAX, ymin, IMAG_MAX, 4096))
            if np.all(np.isfinite(v)) and np.all(v != 0): break
            ymin *= 0.5
        caixa = (X_MIN, X_MAX, ymin, IMAG_MAX); info["imag_min"][l] = ymin
        filtra = lambda zs: [z for z in sem_duplicatas([no_eixo(z) for z in zs]) if ymin <= z.imag < 0 and X_MIN <= z.real <= X_MAX]
        zs = zeros_rapido(F, esp, caixa, prazo)
        if zs is None: zeros_por_l[l] = None; info[l] = "sem_tempo"; continue
        zs = filtra(zs); esperado, ok = auditoria(F, caixa, len(zs)); via = "rapida"
        if not ok:
            a = np.abs(F(xr.astype(complex))); topo = np.flatnonzero((a[1:-1] > a[:-2]) & (a[1:-1] > a[2:])) + 1
            bordas = np.unique(np.r_[X_MIN, xr[topo], X_MAX]); mais = []
            for xa, xb in zip(bordas[:-1], bordas[1:]):
                mais += zeros_na_caixa(F, (xa, xb, ymin, IMAG_MAX), prazo)
                if time.time() >= prazo: break
            if time.time() >= prazo: zeros_por_l[l] = None; info[l] = "sem_tempo"; continue
            zs = filtra(zs + mais); esperado, ok = auditoria(F, caixa, len(zs)); via = "completa"
        zs = [polo_estreito(l, z, delta, mat) for z in zs]
        info[l] = "ok" if ok else f"auditoria_falhou ({esperado} esperados, {len(zs)} achados)"
        zeros_por_l[l] = zs if ok else None       # se a contagem nao fechou, esse l fica com a malha
        polos += [{"l": l, "pole_real": z.real, "pole_imag": z.imag, "Gamma_pole": 2 * abs(z.imag), "Q_pole": z.real / (2 * abs(z.imag)),
                   "argument_count_l": esperado, "argument_audit_ok": ok, "busca": via, "energy_peak_found": False, "xM_energy_peak": np.nan} for z in zs]
    return polos, zeros_por_l, info


# ---------------------------------------------------------------------------
# Uma curva
# ---------------------------------------------------------------------------
def associa_polos(res, polos, esp):
    """Pendura em cada ressonancia os polos que caem dentro dela (o mais forte vira o 'primary')."""
    if not polos:
        for r in res: r.update(primary_l=None, primary_pole_real=np.nan, primary_pole_imag=np.nan, primary_Gamma_pole=np.nan, primary_Q_pole=np.nan, associated_l="", N_associated_Dzeros=0)
        return
    P = np.array([p["pole_real"] for p in polos])
    for r in res:
        tol = max(r["FWHM_energy"] if np.isfinite(r["FWHM_energy"]) else 0, 0.25 * esp)
        perto = [polos[j] for j in np.flatnonzero(np.abs(P - r["xM_peak"]) <= tol)]
        if perto:
            p = min(perto, key=lambda q: abs(q["pole_real"] - r["xM_peak"]))
            r.update(primary_l=p["l"], primary_pole_real=p["pole_real"], primary_pole_imag=p["pole_imag"], primary_Gamma_pole=p["Gamma_pole"], primary_Q_pole=p["Q_pole"],
                     associated_l=",".join(str(l) for l in sorted({q["l"] for q in perto})), N_associated_Dzeros=len(perto))
            for q in perto: q["energy_peak_found"] = True; q["xM_energy_peak"] = r["xM_peak"]
        else:
            r.update(primary_l=None, primary_pole_real=np.nan, primary_pole_imag=np.nan, primary_Gamma_pole=np.nan, primary_Q_pole=np.nan, associated_l="", N_associated_Dzeros=0)


def medir_polos(polos, delta, mat, x, w, esp):
    """Para cada polo: peso S, area pi S/gamma, altura S/gamma^2, fundo de W_T na janela [x_k - esp/2, x_k + esp/2]
    (mediana da malha base, que e' suave e a malha pega bem) e a fracao da energia integrada da janela que vem dele:
        fracao = area_k / (soma das areas dos polos dentro da janela + fundo * largura da janela)."""
    if not polos: return
    for p in polos:
        z = complex(p["pole_real"], p["pole_imag"]); g = -z.imag; S = peso_polo(p["l"], z, delta, mat)
        p.update(S_pole=S, area_pole=np.pi * S / g, WT_peak_pole=S / g**2)
    ordem = np.argsort([p["pole_real"] for p in polos]); X = np.array([polos[i]["pole_real"] for i in ordem])
    G = np.array([-polos[i]["pole_imag"] for i in ordem]); A = np.maximum(np.array([polos[i]["area_pole"] for i in ordem]), 0)
    for p in polos:
        xk = p["pole_real"]; lo, hi = max(xk - 0.5 * esp, X_MIN), min(xk + 0.5 * esp, X_MAX)
        k = (x >= lo) & (x <= hi) & np.isfinite(w); fundo = float(np.median(w[k])) if k.sum() >= 3 else float(np.interp(xk, x, w))
        i0, i1 = np.searchsorted(X, xk - 2 * esp), np.searchsorted(X, xk + 2 * esp)
        Ij = np.sum(A[i0:i1] * (np.arctan((hi - X[i0:i1]) / G[i0:i1]) - np.arctan((lo - X[i0:i1]) / G[i0:i1])) / np.pi)
        I_jan = Ij + max(fundo, 0) * (hi - lo)
        p.update(WT_background_pole=fundo, frac_area_local=float(p["area_pole"] / I_jan) if I_jan > 0 else np.nan,
                 prom_log10_pole=float(np.log10(1 + p["WT_peak_pole"] / fundo)) if fundo > 0 and p["WT_peak_pole"] > 0 else np.nan)


def ressonancia_de_polo(p):
    """Ressonancia lida direto do polo: posicao x_k, FWHM = 2 gamma, Q = x_k / (2 gamma), altura e area analiticas."""
    x, g, fundo = p["pole_real"], -p["pole_imag"], p["WT_background_pole"]; topo = p["WT_peak_pole"] + fundo
    return {"xM_peak": x, "WT_peak": topo, "WT_background": fundo, "energy_ratio": topo / fundo if fundo > 0 else np.inf,
            "energy_log10_prominence": p["prom_log10_pole"], "FWHM_energy": 2 * g, "Q_energy": x / (2 * g), "origem": "polo",
            "area_energy": p["area_pole"], "frac_area_local": p["frac_area_local"], "primary_l": p["l"], "primary_pole_real": x,
            "primary_pole_imag": p["pole_imag"], "primary_Gamma_pole": 2 * g, "primary_Q_pole": x / (2 * g), "associated_l": str(p["l"]), "N_associated_Dzeros": 1}


K_VIS = 10**PROM_WT - 1     # pico visivel: W_T pelo menos 10^PROM_WT acima do fundo

def passo_necessario(p):
    """Maior passo de malha uniforme que garante ver o pico desse polo. No pior caso o ponto mais perto do topo fica a
    dx/2 dele, e ainda assim W_T tem que ficar 10^PROM_WT acima do fundo:
        S / (gamma^2 + (dx/2)^2) >= k fundo   ->   dx_nec = 2 sqrt(S/(k fundo) - gamma^2),   k = 10^PROM_WT - 1.
    (Conferido: a contagem prevista assim acompanha a contagem da malha de 1e-3 ate 1e-6.)"""
    S, fu, g = p.get("S_pole", np.nan), p.get("WT_background_pole", np.nan), abs(p["pole_imag"])
    if not (np.isfinite(S) and np.isfinite(fu) and S > 0 and fu > 0): return np.nan
    return float(2 * np.sqrt(max(S / (K_VIS * fu) - g * g, 0.0)))


def malha_dx(x, alvos, esp, laurent=None, x_max=X_MAX):
    """Malha base + um trecho uniforme em volta de cada grupo de picos (vizinhos a menos de esp/2). O passo do trecho
    e' o do pico mais exigente do grupo (dx_nec) e, se a expansao de Laurent disser que dois picos do grupo estao mais
    perto que isso, metade da separacao deles. Os pontos comecam na borda do trecho, nao em cima do pico, entao o pior
    caso e' o topo ficar a dx/2 do ponto mais proximo, que e' justamente o que dx_nec garante.
    Se nao couber em N_MAX_PONTOS, sobe um piso de passo ate caber (os picos com dx_nec abaixo do piso ficam sem resolver).
    alvos: lista de (posicao, dx_alvo, objeto ou None). Devolve a malha, o piso e grava 'dx_malha_usado' em cada objeto."""
    A = sorted([a for a in alvos if np.isfinite(a[1]) and a[1] > 0], key=lambda a: a[0])
    if not A: return x, np.nan
    P = np.array([a[0] for a in A]); Dn = np.array([a[1] for a in A]); ext = min(esp, 0.05 * x_max)
    seg = []
    for gi in np.split(np.arange(len(P)), np.flatnonzero(np.diff(P) > 0.5 * esp) + 1):
        lo, hi = max(P[gi[0]] - 0.25 * esp, X_MIN), min(P[gi[-1]] + 0.25 * esp, x_max + ext); d = float(Dn[gi].min())
        if laurent is not None and len(laurent[0]):
            k = (laurent[0] >= lo) & (laurent[0] <= hi) & np.isfinite(laurent[1]) & (laurent[1] > 0)
            if k.any(): d = min(d, 0.5 * float(laurent[1][k].min()))
        seg.append((lo, hi, d, gi))
    livre = max(N_MAX_PONTOS - len(x), 1000)
    custo = lambda piso: sum((hi - lo) / max(d, piso) for lo, hi, d, _ in seg)
    piso = DX_MIN
    if custo(piso) > livre:                                   # bissecao em log no piso ate caber
        a, b = DX_MIN, max(hi - lo for lo, hi, _, _ in seg)
        for _ in range(60):
            m = np.sqrt(a * b); a, b = (m, b) if custo(m) > livre else (a, m)
        piso = b
    extras = []
    for lo, hi, d, gi in seg:
        h = max(d, piso); extras.append(lo + h * np.arange(int((hi - lo) / h) + 1))
        for i in gi:
            if A[i][2] is not None: A[i][2]["dx_malha_usado"] = h
    return np.unique(np.concatenate([x] + extras)), float(piso)


def processa_curva(alpha, beta, delta, regime, f, prazo_polos, forcar_polos=False, frac_area=FRAC_AREA):
    t0 = time.time(); mat = materiais(alpha, beta, regime, f)
    x, esp = malha_base(delta, mat); WT = lambda v: energia_total(v, delta, mat)
    risco = alpha < 1 or beta < 1 or forcar_polos          # forcar_polos: so pra teste/calibracao em curvas normais

    # malha comum: sempre roda, e nas curvas normais ja e' a resposta
    w = WT(x); _, xs_base = maximos(WT, x, esp, prom=PROM_WT, w=w)
    saida = {"risco": risco, "candidatos": [], "busca_polos": "nao_feita", "N_res_malha_base": len(xs_base), "N_res_malha_refinada": None,
             "N_modos": None, "pontos_malha": len(x), "dx_min": None}
    if not risco:
        dx = np.interp(xs_base, x[1:], np.diff(x)) if len(xs_base) else None
        res = medir_picos(WT, x, xs_base, esp, escala=dx)
        for r in res: r["origem"] = "malha_base"
        associa_polos(res, [], esp)
        saida.update(modo="malha_base", ressonancias=res, ressonancias_malha=res, runtime_s=time.time() - t0)
        return saida

    # curva de risco: polos, modos por l, malha refinada
    if risco_overflow(alpha, beta, delta, regime):
        zeros_por_l = {l: None for l in range(LMAX + 1)}; saida["busca_polos"] = "pulada_overflow"
    else:
        polos, zeros_por_l, info = buscar_polos(delta, mat, time.time() + prazo_polos, x, esp)
        imag_min = min(info.pop("imag_min").values(), default=np.nan)
        saida.update(candidatos=polos, auditoria=info, imag_min_busca=imag_min, busca_polos="incompleta" if "sem_tempo" in info.values() else "completa")
    modos = modos_por_l(delta, mat, zeros_por_l, x, esp)
    bons = [p for p in saida["candidatos"] if p["argument_audit_ok"]]
    laurent = separacao_laurent(bons, delta, mat) if len(bons) >= 2 else None
    if laurent is not None:
        d_polos = np.diff(np.sort([p["pole_real"] for p in bons]))
        saida.update(N_pares_fundidos_laurent=int(np.sum(np.isnan(laurent[1]))), dx_min_polos=float(0.5 * d_polos.min()),
                     dx_min_laurent=float(0.5 * np.nanmin(laurent[1])) if np.isfinite(laurent[1]).any() else None)

    # malha refinada pelo passo necessario de cada pico (dx_nec, vem dos polos)
    if saida["candidatos"]: medir_polos(saida["candidatos"], delta, mat, x, w, esp)       # S, area, fundo de cada polo
    vis = [p for p in bons if p["Q_pole"] >= Q_MIN_POLO and np.isfinite(p.get("prom_log10_pole", np.nan)) and p["prom_log10_pole"] >= PROM_WT]
    for p in saida["candidatos"]: p["dx_nec"] = passo_necessario(p) if "S_pole" in p else np.nan
    alvos = [(p["pole_real"], p["dx_nec"], p) for p in vis]
    for m in modos:                                   # l sem polos (auditoria falhou): modos da malha, passo = metade da distancia
        if m.get("N_zeros") is None or m["regra"] in ("proeminencia", "sem_polos"):
            P = np.sort(np.asarray(m["x"], float)); viz = np.minimum(np.r_[np.inf, np.diff(P)], np.r_[np.diff(P), np.inf]) if len(P) else []
            alvos += [(v, 0.5 * min(d, 0.25 * esp), None) for v, d in zip(P, viz)]
    xr, piso = malha_dx(x, alvos, esp, laurent); coube = not (np.isfinite(piso) and piso > DX_MIN); dx_min = float(np.min(np.diff(xr)))
    dn = np.array([p["dx_nec"] for p in vis if np.isfinite(p["dx_nec"])])
    saida.update(N_polos_visiveis=len(vis), dx_necessario=float(dn.min()) if len(dn) else None, dx_90=float(np.percentile(dn, 10)) if len(dn) else None,
                 dx_piso_malha=piso if np.isfinite(piso) else None,
                 N_previsto_malha=int(sum(p.get("dx_malha_usado", np.inf) <= p["dx_nec"] for p in vis if np.isfinite(p["dx_nec"]))))
    wr = WT(xr); _, xs = maximos(WT, xr, esp, prom=PROM_WT if coube else None, w=wr)

    # escala de largura pra medir cada pico: Gamma do polo mais proximo, se tiver; senao o passo local da malha
    escala = np.interp(xs, xr[1:], np.diff(xr)) if len(xs) else np.array([])
    if saida["candidatos"] and len(xs):
        P = np.array([p["pole_real"] for p in saida["candidatos"]]); G = np.array([p["Gamma_pole"] for p in saida["candidatos"]])
        j = np.argmin(np.abs(P[None, :] - xs[:, None]), axis=1); perto = np.abs(P[j] - xs) <= 0.25 * esp
        escala = np.where(perto, G[j], escala)
    res = medir_picos(WT, xr, xs, esp, escala=escala)
    for r in res: r["origem"] = "malha_refinada" if coube else "malha_limite"
    associa_polos(res, saida["candidatos"], esp)
    modo = "malha_refinada" if coube else "malha_limite"; saida["xs_malha_refinada"] = [float(v) for v in xs]
    saida["ressonancias_malha"] = list(res)            # picos medidos na malha refinada: ficam salvos mesmo quando a contagem final vem dos polos

    # contagem final pelos polos (nao depende de malha), quando a busca fechou em todos os l
    completa = saida["busca_polos"] == "completa" and all(v == "ok" for v in saida.get("auditoria", {}).values())
    if completa:
        conta = [p for p in sorted(saida["candidatos"], key=lambda q: q["pole_real"]) if np.isfinite(p["frac_area_local"]) and p["frac_area_local"] >= frac_area and p["Q_pole"] >= Q_MIN_POLO]
        ids = {id(p) for p in conta}
        for p in saida["candidatos"]: p["contado_area"] = id(p) in ids
        for p in conta: p["energy_peak_found"] = True; p["xM_energy_peak"] = p["pole_real"]
        # polos sobrepostos: vizinho a menos de (gamma_1 + gamma_2) -> a curva mostra um pico so pros dois.
        # N_res_polos_area conta modos (cada polo); N_res_polos_picos junta os sobrepostos (o que uma malha perfeita veria)
        Xc, Gc = np.array([p["pole_real"] for p in conta]), np.array([-p["pole_imag"] for p in conta])
        junta = np.diff(Xc) < Gc[1:] + Gc[:-1] if len(conta) > 1 else np.array([], bool)
        for i, p in enumerate(conta): p["sobreposto"] = bool((i > 0 and junta[i - 1]) or (i < len(junta) and junta[i]))
        saida.update(N_res_polos_area=len(conta), N_res_polos_picos=int(len(conta) - junta.sum()), area_polos_total=float(sum(p["area_pole"] for p in saida["candidatos"])))
        res, modo = [ressonancia_de_polo(p) for p in conta], "polos_area"
    saida.update(modo=modo, ressonancias=res, N_res_malha_refinada=len(xs), N_modos=sum(len(m["x"]) for m in modos),
                 N_polos_estreitos=sum(-p["pole_imag"] < 1e-15 * p["pole_real"] for p in saida["candidatos"]),
                 regras_l={m["l"]: m["regra"] for m in modos}, pontos_malha=len(xr), dx_min=dx_min, runtime_s=time.time() - t0)
    return saida


class Estourou(Exception): pass

def _alarme(*_): raise Estourou()


def trabalho(args):
    """Roda uma curva dentro de um limite duro de tempo (SIGALRM). Nunca derruba o processo."""
    chave, alpha, beta, delta, regime, f, prazo_polos, limite, frac, norm = args
    global NORMALIZACAO; NORMALIZACAO = norm
    signal.signal(signal.SIGALRM, _alarme); signal.setitimer(signal.ITIMER_REAL, limite)
    try:
        r = processa_curva(alpha, beta, delta, regime, f, prazo_polos, frac_area=frac); r["status"] = "ok"
    except Estourou:
        r = {"status": "timeout"}
    except Exception as e:
        r = {"status": "erro", "erro": f"{type(e).__name__}: {e}"}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    r.update(alpha=alpha, beta=beta, delta=delta, eta=(alpha - beta) / (alpha + beta), regime=regime, f=f)
    return chave, r


# ---------------------------------------------------------------------------
# Checkpoint e CSV
# ---------------------------------------------------------------------------
def caminhos(raiz, regime, f):
    nome = f"coreshell_lmax{LMAX}_{regime}" + (f"_mix{f:.2f}" if regime == "mixed" else "")
    pasta = os.path.join(os.path.abspath(raiz), nome); os.makedirs(pasta, exist_ok=True)
    return {"ckpt": os.path.join(pasta, f"checkpoint_{nome}.npz"), "resumo": os.path.join(pasta, f"resumo_{nome}.csv"),
            "res": os.path.join(pasta, f"ressonancias_{nome}.csv"), "cand": os.path.join(pasta, f"candidatos_{nome}.csv"),
            "res_malha": os.path.join(pasta, f"ressonancias_malha_{nome}.csv"), "leiame": os.path.join(pasta, "LEIAME.txt"),
            "param": os.path.join(pasta, f"parametros_{nome}.json")}


def salva(cam, curvas, alphas, betas, regime, f):
    tmp = cam["ckpt"] + f".tmp_{uuid.uuid4().hex}.npz"
    np.savez_compressed(tmp, versao=VERSAO, regime=regime, f=f, lmax=LMAX, curves=np.array(curvas, dtype=object), alpha_grid=alphas, beta_grid=betas, delta_values=DELTAS)
    os.replace(tmp, cam["ckpt"])


def carrega(cam, regime, f):
    if not os.path.exists(cam["ckpt"]): return {}
    with np.load(cam["ckpt"], allow_pickle=True) as d:
        if int(d["versao"]) != VERSAO or str(d["regime"]) != regime or not np.isclose(float(d["f"]), f):
            print("Checkpoint de outra versao/configuracao, comecando do zero."); return {}
        return d["curves"].item()


LEIAME = """Gerador core-shell fluido, versao {versao}. Arquivos desta pasta:

checkpoint_*.npz        tudo o que foi calculado, curva a curva (dict 'curves'); os CSVs saem dele
parametros_*.json       constantes e opcoes de linha de comando usadas nesta rodada
resumo_*.csv            uma linha por curva
ressonancias_*.csv      uma linha por ressonancia da contagem final (N_res_energy)
ressonancias_malha_*.csv uma linha por pico achado na malha (base nas curvas normais, refinada nas de risco)
candidatos_*.csv        uma linha por polo (zero de D_l) achado na busca

Energia total: W_T = alpha W~_A + beta W~_B (W~_A normalizada por Z_A, W~_B por Z_B; a soma fica normalizada por Z_M).
Com --normalizacao separada volta a soma W~_A + W~_B das versoes antigas (ver parametros_*.json).
Espacamento medio entre ressonancias: Delta_W = pi / (c_M/c_A + delta c_M/c_B); no codigo e' 'esp'. E' a largura da
janela da fracao de area e a escala dos grupos de picos na malha refinada.

Como cada curva e' contada
  alpha >= 1 e beta >= 1: maximos de W_T na malha base (5000 pontos) com proeminencia >= 0,1 em log10.
  alpha < 1 ou beta < 1 (curva de risco):
    1. busca dos zeros de z^4 D_l para l = 0..10 (Newton + principio do argumento; busca completa se nao fechar);
       polos com gamma/x < 1e-10 tem gamma refeito pelo eixo real (D_l = D_j + i D_y);
    2. malha refinada: cada grupo de picos ganha passo dx_nec = 2 sqrt(S/(k fundo) - gamma^2), k = 10^0,1 - 1
       (maior passo que garante ver o pico com o ponto mais proximo a dx/2 do topo), ou metade da separacao de
       Laurent se for menor; teto de {nmax} pontos (acima disso sobe um piso de passo);
    3. se a busca fechou em todos os l: ressonancias = polos com Q >= {qmin} e area >= {frac} da energia da janela
       pi/Phi. Senao: picos da malha refinada.

resumo
  N_res_energy            contagem final (a que vai pro mapa)
  modo_contagem           malha_base | malha_refinada | malha_limite | polos_area
  N_res_malha_base        maximos de W_T na malha base
  N_res_malha_refinada    maximos de W_T na malha refinada (passo dx_nec)
  N_res_polos_area        polos contados (um por modo)
  N_res_polos_picos       idem, juntando polos sobrepostos (|x1 - x2| < gamma1 + gamma2)
  N_polos_visiveis        polos com Q >= 2 e proeminencia analitica >= 0,1 (o que uma malha infinitamente fina veria)
  N_previsto_malha        quantos desses o passo usado deveria ver (dx_malha_usado <= dx_nec)
  frac_malha_refinada     N_res_malha_refinada / N_polos_visiveis
  dx_necessario, dx_90    passo para ver todos / 90% dos polos visiveis
  dx_piso_malha           piso de passo usado na malha refinada (> 1e-9 quando nao coube)
  N_Dzeros_total          polos achados; N_polos_estreitos: polos com gamma/x < 1e-15
  busca_polos, audit_all_ok  se a busca de polos fechou
  demais colunas como nas versoes anteriores (Q_energy_max, energy_ratio_max, x_primeira_res_energy, ...)

ressonancias e ressonancias_malha
  xM_peak, WT_peak, WT_background, energy_ratio, energy_log10_prominence
  FWHM_energy, Q_energy   medidos na malha; nas ressonancias de origem 'polo': FWHM = 2 gamma, Q = x/(2 gamma)
  area_energy             pi S/gamma (so origem 'polo'); frac_area_local: area / energia integrada na janela pi/Phi
  origem                  malha_base | malha_refinada | malha_limite | polo
  primary_l, primary_pole_real, primary_pole_imag, primary_Gamma_pole, primary_Q_pole, associated_l, N_associated_Dzeros

candidatos
  l, pole_real, pole_imag (z = x - i gamma), Gamma_pole = 2 gamma, Q_pole = x/(2 gamma)
  argument_count_l, argument_audit_ok, busca (rapida | completa)
  S_pole                  peso: W_l(x) ~ S/((x - x_k)^2 + gamma^2) perto do polo
  area_pole, WT_peak_pole, WT_background_pole, frac_area_local, prom_log10_pole
  contado_area, sobreposto, dx_nec, dx_malha_usado
"""


def exporta(cam, curvas, args=None, regime=None, f=None):
    resumo, res, cand, res_malha = [], [], [], []
    for chave in sorted(curvas, key=lambda k: tuple(map(int, k.split("_")))):
        c = curvas[chave]; idd, ip = map(int, chave.split("_")); ok = c["status"] == "ok"
        base = {"idd": idd, "ip": ip, **{k: c[k] for k in ("delta", "alpha", "beta", "eta", "regime")}, "mix_factor": c["f"]}
        rr, cc = c.get("ressonancias", []), c.get("candidatos", [])
        Q = [r["Q_energy"] for r in rr if np.isfinite(r["Q_energy"])]; razoes = [r["energy_ratio"] for r in rr if np.isfinite(r["energy_ratio"])]
        aud = c.get("auditoria", {})
        resumo.append({**base, "lmax": LMAX, "status": c["status"],
                       "N_Dzeros_total": len(cc) if ok else np.nan,
                       "N_Dzeros_com_maximo_energia": sum(p["energy_peak_found"] for p in cc) if ok else np.nan,
                       "N_res_energy": len(rr) if ok else np.nan,
                       "N_l_res_energy": len({r["primary_l"] for r in rr if r["primary_l"] is not None}) if ok else np.nan,
                       "has_resonance": bool(rr) if ok else np.nan,
                       "x_primeira_res_energy": rr[0]["xM_peak"] if rr else np.nan,
                       "Q_energy_max": max(Q) if Q else np.nan, "energy_ratio_max": max(razoes) if razoes else np.nan,
                       "audit_all_ok": all(v == "ok" for v in aud.values()) if aud else np.nan,
                       "curva_risco": c.get("risco"), "modo_contagem": c.get("modo"), "busca_polos": c.get("busca_polos"),
                       "N_res_malha_base": c.get("N_res_malha_base"), "N_res_malha_refinada": c.get("N_res_malha_refinada"), "N_res_polos_area": c.get("N_res_polos_area"), "N_res_polos_picos": c.get("N_res_polos_picos"),
                       "N_polos_visiveis": c.get("N_polos_visiveis"), "N_previsto_malha": c.get("N_previsto_malha"),
                       "frac_malha_refinada": (c.get("N_res_malha_refinada") / c["N_polos_visiveis"]) if c.get("N_polos_visiveis") else None,
                       "dx_necessario": c.get("dx_necessario"), "dx_90": c.get("dx_90"), "dx_piso_malha": c.get("dx_piso_malha"),
                       "area_polos_total": c.get("area_polos_total"), "N_polos_estreitos": c.get("N_polos_estreitos"), "N_modos_por_l": c.get("N_modos"),
                       "pontos_malha": c.get("pontos_malha"), "dx_min_refino": c.get("dx_min"),
                       "dx_min_polos": c.get("dx_min_polos"), "dx_min_laurent": c.get("dx_min_laurent"), "N_pares_fundidos_laurent": c.get("N_pares_fundidos_laurent"), "runtime_s": c.get("runtime_s"), "error": c.get("erro", "")})
        res += [{**base, "res_num": i + 1, **r} for i, r in enumerate(rr)]
        res_malha += [{**base, "res_num": i + 1, **r} for i, r in enumerate(c.get("ressonancias_malha", []))]
        cand += [{**base, "candidate_num": i + 1, **p} for i, p in enumerate(cc)]
    for nome, linhas in (("resumo", resumo), ("res", res), ("res_malha", res_malha), ("cand", cand)): pd.DataFrame(linhas).to_csv(cam[nome], index=False)
    with open(cam["leiame"], "w", encoding="utf-8") as fh:
        fh.write(LEIAME.format(versao=VERSAO, nmax=N_MAX_PONTOS, qmin=Q_MIN_POLO, frac=getattr(args, "frac_area", FRAC_AREA)))
    const = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in globals().items() if k.isupper() and isinstance(v, (int, float, str, tuple, np.ndarray)) and k != "LEIAME"}
    with open(cam["param"], "w", encoding="utf-8") as fh:
        json.dump({"versao": VERSAO, "regime": regime, "mix_factor": f, "argumentos": vars(args) if args is not None else None, "constantes": const,
                   "curvas_ok": sum(c["status"] == "ok" for c in curvas.values()), "curvas_total": len(curvas)}, fh, indent=1, default=str)
    print(f"CSVs salvos em {os.path.dirname(cam['resumo'])}")


# ---------------------------------------------------------------------------
# Disparo
# ---------------------------------------------------------------------------
def roda(args, regime, f):
    alphas, betas = grade_eta(args.por_bin, args.seed); cam = caminhos(args.saida, regime, f); curvas = carrega(cam, regime, f)
    pend = [(f"{i}_{p}", float(alphas[p]), float(betas[p]), float(d), regime, f, args.prazo_polos, args.limite_curva, args.frac_area, args.normalizacao)
            for i, d in enumerate(DELTAS) for p in range(len(alphas)) if f"{i}_{p}" not in curvas or (args.refazer_falhas and curvas[f"{i}_{p}"]["status"] != "ok")]
    if args.max_curvas: pend = pend[: args.max_curvas]
    total, prontas = len(alphas) * len(DELTAS), sum(c["status"] == "ok" for c in curvas.values())
    print(f"{regime}" + (f" f={f}" if regime == "mixed" else "") + f": {prontas} de {total} curvas ja feitas, {len(pend)} pra rodar agora com {args.workers} workers")
    t0, feitas = time.time(), 0
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for fut in as_completed([ex.submit(trabalho, t) for t in pend]):
                chave, r = fut.result(); curvas[chave] = r; feitas += 1
                resto = (time.time() - t0) / feitas * (len(pend) - feitas) / 3600
                print(f"[{feitas}/{len(pend)}] {chave} a={r['alpha']:.4g} b={r['beta']:.4g} d={r['delta']:.2f} {r['status']} {r.get('modo', '')} "
                      f"N_res={len(r.get('ressonancias', [])) if r['status'] == 'ok' else 'NA'} (base {r.get('N_res_malha_base', '-')}) polos={len(r.get('candidatos', []))} t={r.get('runtime_s', 0):.1f}s falta~{resto:.1f}h", flush=True)
                if feitas % args.salvar_a_cada == 0: salva(cam, curvas, alphas, betas, regime, f)
    except KeyboardInterrupt:
        print("Interrompido, salvando o que ja foi feito...")
    salva(cam, curvas, alphas, betas, regime, f); exporta(cam, curvas, args, regime, f)


def main():
    ap = argparse.ArgumentParser(description="Ressonancias da esfera core-shell fluida: malha + polos nas curvas de risco.")
    ap.add_argument("--regime", nargs="+", choices=("velocity", "mixed", "density"), required=True)
    ap.add_argument("--mix-factor", nargs="+", type=float, default=[0.5], help="um ou mais f (so vale pro mixed)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--saida", default="./resultados_coreshell")
    ap.add_argument("--prazo-polos", type=float, default=120.0, help="tempo maximo (s) da busca de polos por curva, somando todos os l")
    ap.add_argument("--limite-curva", type=float, default=600.0, help="limite duro (s) por curva")
    ap.add_argument("--salvar-a-cada", type=int, default=25)
    ap.add_argument("--normalizacao", choices=("ZM", "separada"), default=NORMALIZACAO, help="ZM: W_T = alpha W~_A + beta W~_B (padrao); separada: W~_A + W~_B como nas versoes antigas")
    ap.add_argument("--frac-area", type=float, default=FRAC_AREA, help="fracao minima da energia da janela pra um polo contar (curvas de risco)")
    ap.add_argument("--max-curvas", type=int, default=None, help="roda so as primeiras N pendentes (teste)")
    ap.add_argument("--refazer-falhas", action="store_true", help="roda de novo as curvas com timeout ou erro")
    ap.add_argument("--por-bin", type=int, default=20); ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    globals()["NORMALIZACAO"] = args.normalizacao
    for regime in args.regime:
        for f in (args.mix_factor if regime == "mixed" else [0.0]):
            roda(args, regime, f)


if __name__ == "__main__":
    main()
