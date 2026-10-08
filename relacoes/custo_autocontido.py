# ============================================================
# CUSTO COMPUTACIONAL POR CONFIGURAÇÃO (resposta ao revisor 2, ponto 4)
# Código autocontido: só precisa da pasta com os checkpoints (PASTA logo abaixo).
#
# Compara, para cada regime e por configuração (alpha, beta, delta):
#   1. cálculo completo do gerador (malha base de W_T + busca de polos + malha refinada):
#      tempo "runtime_s" que o gerador salvou em cada curva (máquina onde o dataset rodou);
#   2. só a malha base de W_T (5000 pontos, l <= 10) + busca dos máximos, medida AQUI numa amostra:
#      o mínimo que a abordagem direta custa, sem polos nem refino;
#   3. as relações, medidas AQUI:
#        existência: P_res(alpha, beta, delta) para todas as configurações de uma vez;
#        recorrência: a sequência inteira de posições até x_M = 1, a partir do primeiro pico.
# Os coeficientes são os do dataset coreshell_final/resultados_mix (Tabelas 2 e 3).
# Saída: PASTA/custo_computacional/custo_computacional.csv e o resumo impresso.
# ============================================================
import os
import sys
import re
import glob
import time

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.special import spherical_jn, spherical_yn

PASTA = next((a for a in sys.argv[1:] if os.path.isdir(a)), os.environ.get("CORESHELL_DADOS", "/content/drive/MyDrive/coreshell_final/resultados_mix"))   # no PC: python script.py <pasta_com_os_checkpoints>
if PASTA.startswith("/content/drive"):
    from google.colab import drive
    drive.mount("/content/drive")

SAIDA = os.path.join(PASTA, "custo_computacional")
os.makedirs(SAIDA, exist_ok=True)

X_MIN, X_MAX = 1e-4, 1.0
N_AMOSTRA_MALHA = 30        # configurações por regime cronometradas na malha base
N_REPETICOES = 5            # repetições na medida das relações (fica a mediana)
SEMENTE = 42
RHO_REF, C_REF = 1000.0, 1480.0

# regimes: fator de mistura f (rho_i = rho_M r^f, c_i = c_M r^(1-f))
CASES = {
    "velocity": {"f": 0.00, "nome": "Contraste na velocidade"},
    "mixed": {"f": 0.50, "nome": "Contraste misto"},
    "density": {"f": 1.00, "nome": "Contraste na densidade"},
}

# existência (Eq. g): c0..c9
COEF_EXIST = {
    "velocity": [-3.5368632234615807, -1.4722161554416646, -1.163130944690804, 4.246050955653441, 0.1694216135501036,
                 -0.2154035917812324, -0.0606191774534995, 0.2777016064862622, -0.3581962971027315, -1.7796516008395715],
    "mixed": [-4.117083535453357, -1.7442894350383038, -0.8343452623713533, 4.070163490573348, 0.1349603294951578,
              -0.3248938319609638, 0.3697797732216697, 0.3037899322157548, -0.1366742097201914, -1.679595301959711],
    "density": [-5.095345177777789, -2.071119179924473, -0.2372794235570799, 5.124518164501582, 0.0389474467171575,
                -0.4395745635202911, 0.6489470953773018, 0.5182772547489628, 0.2509904003564432, -2.063147445341049],
}

# recorrência ancorada (Eq. recorrência): a0..a5, b0..b5 (não existe para o regime de densidade)
COEF_REC = {
    "velocity": ([0.27885001179193786, -0.29666983480179976, -0.10615391266173438, 0.017848712003096068,
                  0.106078252375186, 3.713049033313154e-05],
                 [1.000243598075701, 0.003616475823146459, -0.003465008822581219, -0.00033086613376530555,
                  -0.0033570597335272363, 0.0020818628450027126]),
    "mixed": ([1.086007074114471, -1.1411652078183805, -0.4412618468853676, 0.05598732693087349,
               0.44045558322973316, 0.00025903701404028645],
              [0.9974431290064801, 0.04689078443117727, -0.037523824430137805, 0.01318779478614604,
               -0.03681783800255972, 0.02182297836168109]),
}


# ============================================================
# CHECKPOINTS
# ============================================================
def achar_checkpoints(pasta):
    """{f: arquivo} para todo checkpoint*mixX.XX*.npz dentro da pasta (e subpastas)."""
    achados = {}
    for arq in sorted(glob.glob(os.path.join(pasta, "**", "*.npz"), recursive=True)):
        nome = os.path.basename(arq)
        if "_dados" in nome or "aleatorio" in nome or not nome.startswith("checkpoint"):
            continue
        m = re.search(r"mix([0-9]+\.[0-9]+)", nome)
        if m:
            achados.setdefault(round(float(m.group(1)), 2), []).append(arq)
    return {f0: sorted(v, key=lambda a: ("reparado" not in a, len(a), a))[0] for f0, v in achados.items()}


def ler_checkpoint(arq):
    """Uma linha por curva válida: alpha, beta, delta, eta, posições dos picos de W_T e runtime_s."""
    with np.load(arq, allow_pickle=True) as data:
        curves = data["curves"].item()
    linhas = []
    for item in curves.values():
        if item.get("status") != "ok":
            continue
        a, b, d = (float(item.get(k, np.nan)) for k in ("alpha", "beta", "delta"))
        if not (np.isfinite(a) and a > 0 and np.isfinite(b) and b > 0 and np.isfinite(d)):
            continue
        picos = sorted(float(p["xM_peak"]) for p in (item.get("ressonancias_malha") or [])
                       if np.isfinite(p.get("xM_peak", np.nan)) and X_MIN <= p["xM_peak"] <= X_MAX)
        rt = item.get("runtime_s")
        linhas.append({"alpha": a, "beta": b, "delta": d, "eta": (a - b) / (a + b), "x_peaks": picos,
                       "runtime_s": float(rt) if rt is not None else np.nan})
    return pd.DataFrame(linhas)


# ============================================================
# ENERGIA INTERNA TOTAL NA MALHA BASE (mesma conta do gerador v8)
# ============================================================
def materiais(alpha, beta, f):
    return (RHO_REF, RHO_REF * alpha**f, RHO_REF * beta**f, C_REF, C_REF * alpha**(1 - f), C_REF * beta**(1 - f))


def energia_total(x, delta, mat, lmax=10):
    """W_T = alpha W~_A + beta W~_B, l = 0..lmax, cada Bessel calculada uma vez por ordem e argumento."""
    rho0, rho1, rho2, c0, c1, c2 = mat
    xA, xB = (c0 / c1) * x, (c0 / c2) * x
    yM, yB = (1 + delta) * x, (1 + delta) * xB
    wA, wB = rho1 * c1 / (rho0 * c0), rho2 * c2 / (rho0 * c0)
    rho_MB, rho_AB, k_MB, k_AB = rho0 / rho2, rho1 / rho2, c2 / c0, c2 / c1

    def jj(n, z):
        return spherical_jn(n, z) if n >= 0 else (-1) ** (-n) * spherical_yn(-n - 1, z)

    def yy(n, z):
        return spherical_yn(n, z) if n >= 0 else (-1) ** (-n - 1) * spherical_jn(-n - 1, z)

    ordens = range(-1, lmax + 2)
    with np.errstate(all="ignore"):
        JA = {n: jj(n, xA) for n in ordens}; JB = {n: jj(n, xB) for n in ordens}; YB = {n: yy(n, xB) for n in ordens}
        Jb = {n: jj(n, yB) for n in ordens}; Yb = {n: yy(n, yB) for n in ordens}
        JM = {n: jj(n, yM) for n in ordens}; YM = {n: yy(n, yM) for n in ordens}
        W = np.zeros_like(x)
        for l in range(lmax + 1):
            d = lambda F, z: F[l - 1] - (l + 1) / z * F[l]
            jA, djA, jB, djB, yB_, dyB = JA[l], d(JA, xA), JB[l], d(JB, xB), YB[l], d(YB, xB)
            jb, djb, yb, dyb = Jb[l], d(Jb, yB), Yb[l], d(Yb, yB)
            h, dh = JM[l] + 1j * YM[l], d(JM, yM) + 1j * d(YM, yM)
            A, B, F, G, HH, JJ = -jb, -yb, rho_MB * h, -djb, -dyb, k_MB * dh
            L, M, P, Q, R, S = -rho_AB * jA, jB, yB_, -k_AB * djA, djB, dyB
            D = (L * R - M * Q) * (F * HH - B * JJ) + (L * S - P * Q) * (A * JJ - F * G)
            X = 1j * rho_MB * k_MB / yM**2 / D
            bl, cl, dl = (M * S - P * R) * X, (P * Q - L * S) * X, (L * R - M * Q) * X
            f_ = 2 * l + 1
            FA_ = 0.5 * xA**3 * (jA**2 - JA[l - 1] * JA[l + 1]); BA_ = xA**2 * jA * djA

            def prim(Jf, Yf, z, jl, yl):
                return (z**3 * (jl**2 - Jf[l - 1] * Jf[l + 1]), z**3 * (yl**2 - Yf[l - 1] * Yf[l + 1]),
                        0.5 * z**3 * (2 * jl * yl - Jf[l + 1] * Yf[l - 1] - Jf[l - 1] * Yf[l + 1]))

            b1, b2, b3 = prim(Jb, Yb, yB, jb, yb); a1, a2, a3 = prim(JB, YB, xB, jB, yB_)
            Fjj, Fyy, Fjy = b1 - a1, b2 - a2, b3 - a3
            Bjj = yB**2 * jb * djb - xB**2 * jB * djB
            Byy = yB**2 * yb * dyb - xB**2 * yB_ * dyB
            Bjy = yB**2 * jb * dyb - xB**2 * jB * dyB
            cl2, dl2, cd = np.abs(cl) ** 2, np.abs(dl) ** 2, np.real(np.conj(cl) * dl)
            w = (wA * np.real(np.abs(bl) ** 2 * f_ * (2 * FA_ + BA_))
                 + wB * np.real(f_ * (cl2 * Fjj + dl2 * Fyy + 2 * cd * Fjy + cl2 * Bjj + dl2 * Byy + 2 * cd * Bjy)))
            W += np.nan_to_num(w)
    return np.where(np.isfinite(W) & (W > 0), W, 0.0)


def malha_base(delta, mat):
    """5000 pontos: 1000 log em [1e-4; 0,1], 3000 lineares em [0,1; 0,9], 1000 lineares em [0,9; 1 + extensão]."""
    rho0, rho1, rho2, c0, c1, c2 = mat
    esp = np.pi / (c0 / c1 + delta * c0 / c2)
    return np.unique(np.concatenate([np.geomspace(X_MIN, 0.1, 1000), np.linspace(0.1, 0.9, 3000),
                                     np.linspace(0.9, X_MAX + min(esp, 0.05), 1000)]))


# ============================================================
# CRONÔMETROS
# ============================================================
def tempo_malha_base(alpha, beta, delta, f):
    """W_T na malha base + máximos com proeminência 0,1 em log10."""
    t0 = time.perf_counter()
    mat = materiais(alpha, beta, f)
    x = malha_base(delta, mat)
    w = energia_total(x, delta, mat)
    ok = w > 0
    if ok.any():
        find_peaks(np.log10(np.where(ok, w, w[ok].min())), prominence=0.1)
    return time.perf_counter() - t0


def tempo_existencia(df, c):
    """P_res para todas as configurações de uma vez; tempo por configuração (mediana das repetições)."""
    a, b, d = (df[k].to_numpy(float) for k in ("alpha", "beta", "delta"))
    tempos = []
    for _ in range(N_REPETICOES):
        t0 = time.perf_counter()
        la, lb = np.log(a), np.log(b)
        g = ((c[0] + c[3] * d + c[9] * d**2) + la * (c[1] + c[6] * d + c[4] * la + c[5] * lb)
             + lb * (c[2] + c[8] * d + c[7] * lb))
        _ = 1.0 / (1.0 + np.exp(-np.clip(g, -700, 700)))
        tempos.append((time.perf_counter() - t0) / len(df))
    return float(np.median(tempos))


def tempo_recorrencia(df, coef, f):
    """Sequência inteira até x_M = 1 a partir do primeiro pico, x_{n+1} = x_ref A + B x_n, x_ref = sqrt(3 alpha^(2-f));
    tempo por configuração ressonante (mediana das repetições)."""
    sub = df.loc[df["x_peaks"].map(len) >= 1]
    if len(sub) == 0:
        return np.nan, 0
    a_, e_, d_ = (sub[k].to_numpy(float) for k in ("alpha", "eta", "delta"))
    x1 = np.array([p[0] for p in sub["x_peaks"]], dtype=float)
    cA, cB = coef
    tempos = []
    for _ in range(N_REPETICOES):
        t0 = time.perf_counter()
        for i in range(len(sub)):
            e, d = e_[i], d_[i]
            termos = (1.0, e, d, e * e, e * d, d * d)
            A = sum(c * v for c, v in zip(cA, termos)); B = sum(c * v for c, v in zip(cB, termos))
            passo = np.sqrt(3.0 * a_[i] ** (2.0 - f)) * A
            x = x1[i]
            for _n in range(100000):
                nx = passo + B * x
                if not (nx > x) or nx > X_MAX:
                    break
                x = nx
        tempos.append((time.perf_counter() - t0) / len(sub))
    return float(np.median(tempos)), len(sub)


def estat(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"mediana": np.nan, "media": np.nan, "p90": np.nan, "total_h": np.nan, "n": 0}
    return {"mediana": float(np.median(v)), "media": float(np.mean(v)), "p90": float(np.percentile(v, 90)),
            "total_h": float(np.sum(v) / 3600), "n": len(v)}


# ============================================================
# RODADA
# ============================================================
GRADES = achar_checkpoints(PASTA)
print("Checkpoints:", {f0: os.path.relpath(a, PASTA) for f0, a in sorted(GRADES.items())})

linhas = []
for regime, caso in CASES.items():
    arq = GRADES.get(round(caso["f"], 2))
    if arq is None:
        print(f"AVISO: sem checkpoint para {caso['nome']}")
        continue
    df = ler_checkpoint(arq)
    risco = (df["alpha"] < 1) | (df["beta"] < 1)

    completo, normal, de_risco = estat(df["runtime_s"]), estat(df.loc[~risco, "runtime_s"]), estat(df.loc[risco, "runtime_s"])
    rng = np.random.default_rng(SEMENTE)
    amostra = df.iloc[rng.choice(len(df), size=min(N_AMOSTRA_MALHA, len(df)), replace=False)]
    malha = estat([tempo_malha_base(r.alpha, r.beta, r.delta, caso["f"]) for r in amostra.itertuples()])
    t_exist = tempo_existencia(df, COEF_EXIST[regime])
    t_rec, n_rec = tempo_recorrencia(df, COEF_REC[regime], caso["f"]) if regime in COEF_REC else (np.nan, 0)

    linha = {"regime": regime, "n_config": len(df), "n_com_runtime": completo["n"],
             "completo_mediana_s": completo["mediana"], "completo_media_s": completo["media"],
             "completo_p90_s": completo["p90"], "completo_total_h": completo["total_h"],
             "completo_normais_mediana_s": normal["mediana"], "completo_risco_mediana_s": de_risco["mediana"],
             "n_normais": normal["n"], "n_risco": de_risco["n"],
             "malha_base_mediana_s": malha["mediana"], "n_amostra_malha": malha["n"],
             "existencia_s": t_exist, "recorrencia_s": t_rec, "n_recorrencia": n_rec}
    for nome, t in (("existencia", t_exist), ("recorrencia", t_rec)):
        linha[f"ganho_{nome}_sobre_malha"] = malha["mediana"] / t if np.isfinite(t) and t > 0 else np.nan
        linha[f"ganho_{nome}_sobre_completo"] = completo["mediana"] / t if np.isfinite(t) and t > 0 else np.nan
    linhas.append(linha)

    print(f"\n{caso['nome']} ({len(df)} configurações):")
    if completo["n"] == 0:
        print("  AVISO: o checkpoint não tem 'runtime_s'; o custo do cálculo completo fica de fora.")
    print(f"  cálculo completo (dataset): mediana {completo['mediana']:.3g} s | média {completo['media']:.3g} s | "
          f"p90 {completo['p90']:.3g} s | total {completo['total_h']:.3g} h")
    print(f"     curvas normais (alpha >= 1 e beta >= 1, {normal['n']}): mediana {normal['mediana']:.3g} s")
    print(f"     curvas de risco (alpha < 1 ou beta < 1, {de_risco['n']}): mediana {de_risco['mediana']:.3g} s")
    print(f"  malha base de W_T (aqui, {malha['n']} configurações): mediana {malha['mediana']:.3g} s")
    print(f"  existência P_res (aqui): {t_exist:.3g} s por configuração")
    if regime in COEF_REC:
        print(f"  recorrência (aqui, {n_rec} configurações ressonantes): {t_rec:.3g} s por configuração")
    else:
        print("  recorrência: não existe para este regime")
    for nome, rotulo in (("existencia", "existência"), ("recorrencia", "recorrência")):
        gm, gc = linha[f"ganho_{nome}_sobre_malha"], linha[f"ganho_{nome}_sobre_completo"]
        if np.isfinite(gm):
            print(f"  ganho da {rotulo}: {gm:.3g}x sobre a malha base" + (f", {gc:.3g}x sobre o cálculo completo" if np.isfinite(gc) else ""))

pd.DataFrame(linhas).to_csv(os.path.join(SAIDA, "custo_computacional.csv"), index=False)
print(f"\nSalvo em: {os.path.join(SAIDA, 'custo_computacional.csv')}")
