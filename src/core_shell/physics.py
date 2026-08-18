import numpy as np
from scipy.special import spherical_jn, spherical_yn


def J(n, x):
    if n >= 0:
        return spherical_jn(n, x)
    m = -n - 1
    return (-1) ** (m + 1) * spherical_yn(m, x)


def Y(n, x):
    if n >= 0:
        return spherical_yn(n, x)
    m = -n - 1
    return (-1) ** m * spherical_jn(m, x)


def dJ(n, x):
    if n >= 0:
        return spherical_jn(n, x, derivative=True)
    m = -n - 1
    return (-1) ** (m + 1) * spherical_yn(m, x, derivative=True)


def dY(n, x):
    if n >= 0:
        return spherical_yn(n, x, derivative=True)
    m = -n - 1
    return (-1) ** m * spherical_jn(m, x, derivative=True)


def H(n, x):
    if n >= 0:
        return spherical_jn(n, x) + 1j * spherical_yn(n, x)
    m = -n - 1
    return 1j * (-1) ** m * (spherical_jn(m, x) + 1j * spherical_yn(m, x))


def dH(n, x):
    if n >= 0:
        return spherical_jn(n, x, derivative=True) + 1j * spherical_yn(n, x, derivative=True)
    m = -n - 1
    return 1j * (-1) ** m * (
        spherical_jn(m, x, derivative=True)
        + 1j * spherical_yn(m, x, derivative=True)
    )


def Fjj(l, xB, yB):
    def F(x):
        return x**3 * (J(l, x) ** 2 - J(l - 1, x) * J(l + 1, x))
    return F(yB) - F(xB)


def Fyy(l, xB, yB):
    def F(x):
        return x**3 * (Y(l, x) ** 2 - Y(l - 1, x) * Y(l + 1, x))
    return F(yB) - F(xB)


def Fjy(l, xB, yB):
    def F(x):
        return 0.5 * x**3 * (
            2 * J(l, x) * Y(l, x)
            - J(l + 1, x) * Y(l - 1, x)
            - J(l - 1, x) * Y(l + 1, x)
        )
    return F(yB) - F(xB)


def Bjj(l, xB, yB):
    def B(x):
        return x**2 * J(l, x) * dJ(l, x)
    return B(yB) - B(xB)


def Byy(l, xB, yB):
    def B(x):
        return x**2 * Y(l, x) * dY(l, x)
    return B(yB) - B(xB)


def Bjy(l, xB, yB):
    def B(x):
        return x**2 * J(l, x) * dY(l, x)
    return B(yB) - B(xB)


def FA(l, xA):
    return 0.5 * xA**3 * (J(l, xA) ** 2 - J(l - 1, xA) * J(l + 1, xA))


def BA(l, xA):
    return xA**2 * J(l, xA) * dJ(l, xA)


def core_shell_denominator(l, xA, xB, yM, yB, rho0, rho1, rho2, c0, c1, c2):
    rho_MB = rho0 / rho2
    rho_AB = rho1 / rho2
    k_MB = c2 / c0
    k_AB = c2 / c1

    A = -J(l, yB)
    B = -Y(l, yB)
    F = rho_MB * H(l, yM)
    G = -dJ(l, yB)
    HH = -dY(l, yB)
    JJ = k_MB * dH(l, yM)
    L = -rho_AB * J(l, xA)
    M = J(l, xB)
    P = Y(l, xB)
    Q = -k_AB * dJ(l, xA)
    R = dJ(l, xB)
    S = dY(l, xB)

    return (L * R - M * Q) * (F * HH - B * JJ) + (L * S - P * Q) * (A * JJ - F * G)


def coefficients(l, xA, xB, yM, yB, rho0, rho1, rho2, c0, c1, c2):
    rho_MB = rho0 / rho2
    rho_AB = rho1 / rho2
    k_MB = c2 / c0
    k_AB = c2 / c1

    A = -J(l, yB)
    B = -Y(l, yB)
    F = rho_MB * H(l, yM)
    G = -dJ(l, yB)
    HH = -dY(l, yB)
    JJ = k_MB * dH(l, yM)
    L = -rho_AB * J(l, xA)
    M = J(l, xB)
    P = Y(l, xB)
    Q = -k_AB * dJ(l, xA)
    R = dJ(l, xB)
    S = dY(l, xB)

    denominator = (L * R - M * Q) * (F * HH - B * JJ) + (L * S - P * Q) * (A * JJ - F * G)
    X = 1j * rho_MB * k_MB / yM**2

    b_l = (M * S - P * R) * X / denominator
    c_l = (P * Q - L * S) * X / denominator
    d_l = (L * R - M * Q) * X / denominator
    return b_l, c_l, d_l


def total_energy_parts_vs_xm(
    xM_values,
    delta,
    rho0,
    rho1,
    rho2,
    c0,
    c1,
    c2,
    lmax_core=10,
    lmax_shell=10,
):
    xM_values = np.asarray(xM_values, dtype=float)
    lmax_core = int(lmax_core)
    lmax_shell = int(lmax_shell)
    lmax_total = max(lmax_core, lmax_shell)

    WK_core = np.zeros_like(xM_values, dtype=float)
    WP_core = np.zeros_like(xM_values, dtype=float)
    WK_shell = np.zeros_like(xM_values, dtype=float)
    WP_shell = np.zeros_like(xM_values, dtype=float)

    xA = (c0 / c1) * xM_values
    xB = (c0 / c2) * xM_values
    yM = (1.0 + delta) * xM_values
    yB = (1.0 + delta) * xB

    with np.errstate(all="ignore"):
        for l in range(lmax_total + 1):
            b_l, c_l, d_l = coefficients(
                l, xA, xB, yM, yB, rho0, rho1, rho2, c0, c1, c2
            )
            factor = 2 * l + 1

            if l <= lmax_core:
                b2 = np.abs(b_l) ** 2
                fa = FA(l, xA)
                ba = BA(l, xA)
                WP_core += np.real(b2 * factor * fa)
                WK_core += np.real(b2 * factor * (fa + ba))

            if l <= lmax_shell:
                c2_abs = np.abs(c_l) ** 2
                d2_abs = np.abs(d_l) ** 2
                re_cd = np.real(np.conj(c_l) * d_l)
                potential = (
                    0.5 * c2_abs * Fjj(l, xB, yB)
                    + 0.5 * d2_abs * Fyy(l, xB, yB)
                    + re_cd * Fjy(l, xB, yB)
                )
                boundary = (
                    c2_abs * Bjj(l, xB, yB)
                    + d2_abs * Byy(l, xB, yB)
                    + 2.0 * re_cd * Bjy(l, xB, yB)
                )
                WP_shell += np.real(factor * potential)
                WK_shell += np.real(factor * (potential + boundary))

    return WK_core, WP_core, WK_shell, WP_shell


def total_internal_energy(
    xM_values,
    delta,
    rho0,
    rho1,
    rho2,
    c0,
    c1,
    c2,
    lmax=10,
):
    parts = total_energy_parts_vs_xm(
        xM_values,
        delta,
        rho0,
        rho1,
        rho2,
        c0,
        c1,
        c2,
        lmax_core=lmax,
        lmax_shell=lmax,
    )
    total = sum(parts)
    return np.where(np.isfinite(total) & (total > 0.0), total, 0.0)


RHO_WATER_REF = 1000.0
C_WATER_REF = 1480.0
Z_WATER = RHO_WATER_REF * C_WATER_REF


def make_materials_from_alpha_beta(
    alpha,
    beta,
    ZM=Z_WATER,
    rho_ref=RHO_WATER_REF,
    c_ref=C_WATER_REF,
    contrast_in="mixed",
    mix_factor=0.5,
):
    alpha = float(alpha)
    beta = float(beta)
    if alpha <= 0.0 or beta <= 0.0:
        raise ValueError("alpha and beta must be positive.")

    ZA = alpha * ZM
    ZB = beta * ZM

    if contrast_in == "sound_speed":
        rho0 = rho1 = rho2 = rho_ref
        c0 = ZM / rho0
        c1 = ZA / rho1
        c2 = ZB / rho2
    elif contrast_in == "density":
        c0 = c1 = c2 = c_ref
        rho0 = ZM / c0
        rho1 = ZA / c1
        rho2 = ZB / c2
    elif contrast_in == "mixed":
        mix_factor = float(mix_factor)
        rho0 = rho_ref
        c0 = ZM / rho0
        rho1 = rho_ref * alpha**mix_factor
        c1 = c_ref * alpha ** (1.0 - mix_factor)
        rho2 = rho_ref * beta**mix_factor
        c2 = c_ref * beta ** (1.0 - mix_factor)
    else:
        raise ValueError('contrast_in must be "sound_speed", "density", or "mixed".')

    return rho0, rho1, rho2, c0, c1, c2


def eta_from_alpha_beta(alpha, beta):
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    return (alpha - beta) / (alpha + beta)
