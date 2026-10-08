# Integral acumulada de |b_l(x)|^2, |c_l(x)|^2 e |d_l(x)|^2 numa malha de elementos adaptada aos polos (FEM + Gauss-Legendre).
# Arquivo autocontido: so usa numpy e scipy. Tudo que entra na integral esta definido aqui.
import json, sys, numpy as np, warnings; warnings.filterwarnings("ignore")
from scipy.special import spherical_jn, spherical_yn
from scipy.optimize import newton
from scipy.integrate import cumulative_trapezoid

# ---------------------------------------------------------------- 1. Bessel esfericas (l >= 0; aceitam argumento complexo)
J  = lambda l, x: spherical_jn(l, x)
Y  = lambda l, x: spherical_yn(l, x)
dJ = lambda l, x: spherical_jn(l, x, derivative=True)
dY = lambda l, x: spherical_yn(l, x, derivative=True)
H  = lambda l, x: J(l, x) + 1j * Y(l, x)            # Hankel de primeira especie h_l = j_l + i y_l
dH = lambda l, x: dJ(l, x) + 1j * dY(l, x)

# ---------------------------------------------------------------- 2. Materiais: (rho_M, rho_A, rho_B, c_M, c_A, c_B)
RHO, C = 1000.0, 1480.0                               # meio M = agua
def materiais(alpha, beta, regime, f=0.5):
    """alpha = Z_A/Z_M, beta = Z_B/Z_M. velocity: so muda a velocidade; mixed: rho ~ alpha^f e c ~ alpha^(1-f)."""
    if regime == "velocity": return RHO, RHO, RHO, C, alpha * C, beta * C
    if regime == "density":  return RHO, alpha * RHO, beta * RHO, C, C, C
    return RHO, RHO * alpha**f, RHO * beta**f, C, C * alpha**(1 - f), C * beta**(1 - f)

# ---------------------------------------------------------------- 3. Argumentos (x = k_M a_A, raio do nucleo a_A, casca ate a_B = (1+delta) a_A)
def argumentos(x, delta, mat):
    rM, rA, rB, cM, cA, cB = mat
    xA = (cM / cA) * x                 # k_A a_A
    xB = (cM / cB) * x                 # k_B a_A
    yM = (1 + delta) * x               # k_M a_B
    yB = (1 + delta) * xB              # k_B a_B
    return xA, xB, yM, yB

# ---------------------------------------------------------------- 4. Sistema de contorno e coeficientes
# Campos: nucleo p_A = b_l j_l(k_A r); casca p_B = c_l j_l(k_B r) + d_l y_l(k_B r); fora p_M = j_l(k_M r) + s_l h_l(k_M r).
# Linhas (continuidade de pressao e de velocidade normal, normalizadas pela casca):
#   r = a_A:  L b + M c + P d         = 0           L = -(rho_A/rho_B) j_l(xA),  M = j_l(xB),   P = y_l(xB)
#             Q b + R c + S d         = 0           Q = -(c_B/c_A) j_l'(xA),     R = j_l'(xB),  S = y_l'(xB)
#   r = a_B:  A c + B d + F s = -(rho_M/rho_B) j_l(yM)        A = -j_l(yB),  B = -y_l(yB),  F = (rho_M/rho_B) h_l(yM)
#             G c + HH d + JJ s = -(c_B/c_M) j_l'(yM)         G = -j_l'(yB), HH = -y_l'(yB), JJ = (c_B/c_M) h_l'(yM)
def blocos(l, x, delta, mat):
    rM, rA, rB, cM, cA, cB = mat; xA, xB, yM, yB = argumentos(x, delta, mat)
    rho_MB, rho_AB, k_MB, k_AB = rM / rB, rA / rB, cB / cM, cB / cA
    A, B, F = -J(l, yB), -Y(l, yB), rho_MB * H(l, yM)
    G, HH, JJ = -dJ(l, yB), -dY(l, yB), k_MB * dH(l, yM)
    L, M, P = -rho_AB * J(l, xA), J(l, xB), Y(l, xB)
    Q, R, S = -k_AB * dJ(l, xA), dJ(l, xB), dY(l, xB)
    return dict(A=A, B=B, F=F, G=G, HH=HH, JJ=JJ, L=L, M=M, P=P, Q=Q, R=R, S=S, rho_MB=rho_MB, k_MB=k_MB, yM=yM)

def D_l(l, x, delta, mat):
    """Denominador comum (determinante do sistema, a menos de constante):
       D_l = (L R - M Q)(F HH - B JJ) + (L S - P Q)(A JJ - F G).   Os polos de b, c, d sao os zeros de D_l."""
    k = blocos(l, np.asarray(x, dtype=complex), delta, mat)
    return (k["L"] * k["R"] - k["M"] * k["Q"]) * (k["F"] * k["HH"] - k["B"] * k["JJ"]) + (k["L"] * k["S"] - k["P"] * k["Q"]) * (k["A"] * k["JJ"] - k["F"] * k["G"])

def numeradores(l, x, delta, mat):
    """Regra de Cramer, usando o Wronskiano j_l h_l' - j_l' h_l = i / yM^2 para simplificar:
         X   = i (rho_M/rho_B)(c_B/c_M) / yM^2
         b_l = (M S - P R) X / D_l,   c_l = (P Q - L S) X / D_l,   d_l = (L R - M Q) X / D_l"""
    k = blocos(l, np.asarray(x, dtype=complex), delta, mat); X = 1j * k["rho_MB"] * k["k_MB"] / k["yM"] ** 2
    N = {"b": (k["M"] * k["S"] - k["P"] * k["R"]) * X, "c": (k["P"] * k["Q"] - k["L"] * k["S"]) * X, "d": (k["L"] * k["R"] - k["M"] * k["Q"]) * X}
    return N, D_l(l, x, delta, mat)

def integrando(l, x, delta, mat):
    """As tres funcoes integradas: |b_l(x)|^2, |c_l(x)|^2, |d_l(x)|^2."""
    N, D = numeradores(l, x, delta, mat); return {k: np.abs(N[k] / D) ** 2 for k in "bcd"}

def confere_contorno(l, x, delta, mat):
    """Conferencia independente: monta o sistema 4x4 acima, resolve com np.linalg.solve e compara com b, c, d da formula fechada."""
    k = blocos(l, complex(x), delta, mat)
    Mx = np.array([[k["L"], k["M"], k["P"], 0], [k["Q"], k["R"], k["S"], 0], [0, k["A"], k["B"], k["F"]], [0, k["G"], k["HH"], k["JJ"]]], complex)
    rhs = np.array([0, 0, -k["rho_MB"] * J(l, k["yM"]), -k["k_MB"] * dJ(l, k["yM"])], complex)
    b, c, d, s = np.linalg.solve(Mx, rhs); N, D = numeradores(l, x, delta, mat)
    return max(abs(b - N["b"] / D) / abs(b), abs(c - N["c"] / D) / abs(c), abs(d - N["d"] / D) / max(abs(d), 1e-300) if abs(d) > 1e-14 * abs(c) else 0.0)

# ---------------------------------------------------------------- 5. Polos e residuos
def D_partes(l, x, delta, mat):
    """No eixo real so F e JJ sao complexos (vem de h_l = j_l + i y_l fora da esfera) e D_l e linear em (F, JJ). Entao
       D_l(x) = D_j(x) + i D_y(x),  D_j = D_l com h_l -> j_l,  D_y = D_l com h_l -> y_l  (os dois reais, calculados separados)."""
    k = blocos(l, np.asarray(x, dtype=float).astype(complex), delta, mat)
    C1, C2 = k["L"] * k["R"] - k["M"] * k["Q"], k["L"] * k["S"] - k["P"] * k["Q"]
    D = lambda F, JJ: np.real(C1 * (F * k["HH"] - k["B"] * JJ) + C2 * (k["A"] * JJ - F * k["G"]))
    return D(k["rho_MB"] * J(l, k["yM"]), k["k_MB"] * dJ(l, k["yM"])), D(k["rho_MB"] * Y(l, k["yM"]), k["k_MB"] * dY(l, k["yM"]))

GAMMA_RAD = 1e-10   # gamma/x abaixo disso o Newton em complexo ja nao resolve gamma (Im de j_l(x + i eps) vira ruido de ~1e-20)
def polo_estreito(l, z, delta, mat):
    """Polo colado no eixo: Newton feito so com valores no eixo real, onde D_j e D_y tem precisao relativa do double.
       D_l(x0 - i gamma) ~ D_l(x0) - i gamma D_l'(x0) = 0   ->   z0 = x0 - D_l(x0) / D_l'(x0),  com D_l = D_j + i D_y.
       Itera x0 <- x0 - Re(D/D') ate Re(D/D') = 0; ai gamma = Im(D_l(x0) / D_l'(x0)).
       Caso limite (radiacao fraca, D_y(x0) = 0): gamma = -D_j(x0) / D_y'(x0)."""
    x, gm = z.real, -z.imag
    if gm > GAMMA_RAD * x: return z
    Dc = lambda t: (lambda p: p[0] + 1j * p[1])(D_partes(l, t, delta, mat))
    for _ in range(50):
        h = 1e-7 * x; q = Dc(x) / ((Dc(x + h) - Dc(x - h)) / (2 * h)); x -= q.real
        if abs(q.real) < 1e-15 * x: break
    return complex(x, -q.imag) if q.imag > 0 else z

def polos(l, delta, mat, a, b, n=400001):
    """Zeros z_k = x_k - i gamma_k de D_l com x_k em [a, b]. Sementes = minimos de |z^4 D_l| no eixo real
    (o z^4 tira o polo de ordem 4 de D_l em z = 0); Newton (secante) no plano complexo a partir de cada semente."""
    F = lambda z: z**4 * D_l(l, z, delta, mat)
    x = np.linspace(a, b, n); v = np.abs(F(x)); sem = x[np.flatnonzero((v[1:-1] < v[:-2]) & (v[1:-1] < v[2:])) + 1]; zs = []
    for s0 in sem:
        cand = []
        for y0 in (-1e-12, -1e-9, -1e-6):
            try: z = complex(newton(F, complex(s0, y0), tol=1e-15, maxiter=200))
            except Exception: continue
            h = 1e-7 * abs(z); viz = np.abs(F(z + h * np.array([1, -1, 1j, -1j])))
            # zero de verdade (nao so um minimo raso). Im(z) > 0 so e aceito se for ruido: |Im z| < 1e-15 |z| (gamma abaixo da precisao do double)
            if a <= z.real <= b and -0.5 <= z.imag < 1e-15 * abs(z) and abs(F(z)) < 1e-7 * np.median(viz):
                cand.append(z)
                if z.imag < 0: break
        if cand:
            z = next((c for c in cand if c.imag < 0), cand[0]); z = complex(z.real, -max(abs(z.imag), 1e-300))
            if all(abs(z - w) > 1e-12 * abs(z) for w in zs): zs.append(z)
    return np.array(sorted([polo_estreito(l, z, delta, mat) for z in zs], key=lambda z: z.real))

def conta_zeros(l, delta, mat, a, b, n=40000):
    """Principio do argumento: N = (1/2 pi i) contorno de D_l'/D_l = numero de voltas da fase de z^4 D_l no retangulo
       [a, b] x [y_baixo, 1e-3]. Confere se a lista de polos do Newton esta completa."""
    t = np.linspace(0, 1, n, endpoint=False)
    for yb in (-0.05, -0.01):          # no velocity muito contrastado o fundo -0.05 estoura; ai usa um mais raso
        ya = 1e-3; c = np.concatenate([a + (b - a) * t + 1j * yb, b + 1j * (yb + (ya - yb) * t), b - (b - a) * t + 1j * ya, a + 1j * (ya - (ya - yb) * t)])
        v = c**4 * D_l(l, c, delta, mat); w = np.sum(np.angle(np.roll(v, -1) / v)) / (2 * np.pi)
        if np.isfinite(w): return int(round(w))
    return -1

def residuos(l, zs, delta, mat):
    """Perto de z_k:  k_l(x) ~ R_k / (x - z_k),  R_k = N_k(z_k) / D_l'(z_k)  (D_l' por diferenca central).
       Entao |k_l(x)|^2 ~ |R_k|^2 / ((x - x_k)^2 + gamma_k^2), uma lorentziana de area pi |R_k|^2 / gamma_k."""
    R = {k: [] for k in "bcd"}
    for z in zs:
        h = 1e-7 * abs(z); N, _ = numeradores(l, z, delta, mat); dD = (D_l(l, z + h, delta, mat) - D_l(l, z - h, delta, mat)) / (2 * h)
        for k in "bcd": R[k].append(N[k] / dD)
    return {k: np.array(v) for k, v in R.items()}

# ---------------------------------------------------------------- 6. Malha de elementos e quadratura
GL_X, GL_W = np.polynomial.legendre.leggauss(8)      # Gauss-Legendre de 8 pontos por elemento (exata para polinomio de grau 15)
EPS_REL = 1e-9   # se gamma_k / x_k < 1e-9, perto do polo o D_l(x) ja e ruido de arredondamento: a lorentziana entra analitica

def bordas(a, b, zs, m, n_fundo):
    """Bordas dos elementos: n_fundo elementos uniformes (fundo) + m elementos por polo com
       x = x_k + gamma_k tan(theta), theta uniforme entre arctan((a-x_k)/gamma_k) e arctan((b-x_k)/gamma_k).
       Cada elemento leva a mesma fracao da area da lorentziana, entao o pico fica sempre bem resolvido, por menor que seja gamma_k."""
    E = [np.linspace(a, b, n_fundo + 1)]
    for z in zs:
        x0, gm = z.real, abs(z.imag)
        if gm < EPS_REL * x0: continue
        th = np.linspace(np.arctan((a - x0) / gm), np.arctan((b - x0) / gm), m + 1); E.append(x0 + gm * np.tan(th))
    E = np.unique(np.clip(np.concatenate(E), a, b)); return E[np.r_[True, np.diff(E) > 0]]

def integra_fem(l, delta, mat, a, b, zs, R, m=32, n_fundo=800):
    """I_k(x) = integral de a ate x de |k_l(t)|^2 dt, nas bordas dos elementos, para k = b, c, d.
       Em cada elemento [e_i, e_i+1]: soma_j w_j f(c_i + h_i t_j / 2) h_i / 2.
       Polo fino (gamma/x < EPS_REL): integra f - |R|^2/((x-x_k)^2+gamma^2) e soma a area analitica |R|^2/gamma [arctan]."""
    E = bordas(a, b, zs, m, n_fundo); h = np.diff(E); c = 0.5 * (E[1:] + E[:-1])
    X = (c[:, None] + 0.5 * h[:, None] * GL_X[None, :]).ravel(); f = integrando(l, X, delta, mat)
    fino = np.array([abs(z.imag) < EPS_REL * z.real for z in zs], bool); I = {}
    for k in "bcd":
        g = f[k].copy(); Ian = np.zeros(len(E))
        for z, Rk in zip(zs[fino], R[k][fino]):
            x0, gm, A2 = z.real, abs(z.imag), abs(Rk) ** 2
            g -= A2 / ((X - x0) ** 2 + gm**2); Ian += A2 / gm * (np.arctan((E - x0) / gm) - np.arctan((a - x0) / gm))
        I[k] = np.r_[0, np.cumsum((g.reshape(len(h), -1) * GL_W).sum(1) * 0.5 * h)] + Ian
    return E, I, len(X)

def acumulada_residuos(x, a, zs, R, k):
    """Soma das lorentzianas integradas: sum_k |R_k|^2/gamma_k [arctan((x-x_k)/gamma_k) - arctan((a-x_k)/gamma_k)]."""
    if not len(zs): return 0 * x
    g = np.abs(zs.imag); A2 = np.abs(R[k]) ** 2
    return np.sum(A2 / g * (np.arctan((x[:, None] - zs.real) / g) - np.arctan((a - zs.real) / g)), axis=1)

# ---------------------------------------------------------------- 7. Curvas testadas: (nome, regime, alpha, beta, delta, x0, n_esp); janela [x0, x0 + n_esp pi/Phi]
CASOS = [("nao patologica 1", "mixed", 0.72222, 36.667, 1.0, 0.55, 3), ("nao patologica 2", "mixed", 0.0063333, 0.001, 0.5, 0.30, 4),
         ("nao patologica 3", "velocity", 0.081111, 0.072222, 0.10, 0.30, 6), ("patologica 1", "velocity", 0.001, 0.001, 1.5, 0.5, 6),
         ("patologica 3", "velocity", 0.0018889, 0.001, 1.5, 0.5, 6), ("patologica 4", "velocity", 0.001, 0.0018889, 1.5, 0.5, 6),
         ("patologica 5", "velocity", 0.0063333, 0.001, 0.5, 0.5, 6), ("patologica 6", "velocity", 0.0063333, 0.001, 1.35, 0.5, 6),
         ("patologica 7", "velocity", 0.001, 0.027778, 1.5, 0.08, 6), ("patologica 8", "velocity", 0.0036667, 0.001, 1.5, 0.5, 6),
         ("patologica 9", "velocity", 0.001, 0.001, 0.8, 0.7, 6), ("patologica 10", "mixed", 0.0063333, 0.001, 1.35, 0.45, 6)]

def roda(nome, regime, alpha, beta, delta, x0, n_esp, l):
    mat = materiais(alpha, beta, regime); rM, rA, rB, cM, cA, cB = mat
    esp = np.pi / (cM / cA + delta * cM / cB); a, b = x0, min(x0 + n_esp * esp, 1.0)       # pi/Phi: espacamento medio dos zeros de um l
    erro_contorno = max(confere_contorno(l, x, delta, mat) for x in np.linspace(a, b, 7))
    zs = polos(l, delta, mat, a, b); R = residuos(l, zs, delta, mat); N_contorno = conta_zeros(l, delta, mat, a, b)
    E1, I1, n1 = integra_fem(l, delta, mat, a, b, zs, R, 64, 1600)                        # malha grossa
    E2, I2, n2 = integra_fem(l, delta, mat, a, b, zs, R, 128, 3200)                       # malha fina (2x mais elementos)
    xt = np.linspace(a, b, 5000); ft = integrando(l, xt, delta, mat)                      # trapezio uniforme, 5000 pontos (como antes)
    xr = np.linspace(a, b, 3000); passo = max(1, len(E2) // 3000)
    out = dict(nome=nome, regime=regime, alpha=alpha, beta=beta, delta=delta, l=l, janela=[a, b], erro_contorno=float(erro_contorno),
               N_contorno=N_contorno, polos=zs.real.tolist(), gamma=np.abs(zs.imag).tolist(), pontos_fem=[n1, n2], coef={})
    for k in "bcd":
        It = cumulative_trapezoid(ft[k], xt, initial=0)
        out["coef"][k] = dict(fem_grossa=float(I1[k][-1]), fem_fina=float(I2[k][-1]), trap=float(It[-1]),
                              res=float(np.sum(np.pi * np.abs(R[k]) ** 2 / np.abs(zs.imag))) if len(zs) else 0.0,
                              xf=E2[::passo].tolist(), If=I2[k][::passo].tolist(), xt=xt[::5].tolist(), It=It[::5].tolist(),
                              xr=xr.tolist(), Ir=acumulada_residuos(xr, a, zs, R, k).tolist())
    c = out["coef"]; print(f"{nome:17s} l={l} polos={len(zs):3d} contorno={N_contorno:3d} erro contorno={erro_contorno:.1e} | " + " | ".join(
        f"{k}: fem {c[k]['fem_fina']:.5g} (grossa-fina {abs(c[k]['fem_grossa'] - c[k]['fem_fina']) / max(abs(c[k]['fem_fina']), 1e-300):.0e}) trap {c[k]['trap']:.4g} res {c[k]['res']:.4g}" for k in "bcd"), flush=True)
    return out

if __name__ == "__main__":
    ls = [int(v) for v in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0, 1]; saida = sys.argv[2] if len(sys.argv) > 2 else "fem_auto.json"
    json.dump([roda(*caso, l) for caso in CASOS for l in ls], open(saida, "w"))
