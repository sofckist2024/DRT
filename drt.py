"""
Distribution of Relaxation Times (DRT) core.

Physics (Choi et al., JOM 71, 3825 (2019)):

    Z_pol(w) = R_pol * integral_0^inf  gamma(tau) / (1 + jw tau) d tau ,
               with  integral gamma(tau) d tau = 1.

This is a Fredholm integral equation of the first kind (ill-posed); it is
solved by Tikhonov regularization with a non-negativity constraint on the
distribution.  Discretising the distribution on a logarithmic tau grid,

    g_m  =  R_pol * gamma(tau_m) * tau_m            (units: Ohm per unit ln tau)

so that the polarization resistance carried by a peak is its area under the
g(ln tau) curve, and

    Z_pol(w_n) = R_inf  +  sum_m  g_m * dlntau / (1 + j w_n tau_m).

Stacking the real and imaginary parts gives a real linear system  A x = b,
x = [g_0 ... g_{M-1}, R_inf] >= 0, solved for each regularization parameter
lambda (= k_reg) by non-negative least squares on the augmented matrix

    minimize || [A ; lambda L] x  -  [b ; 0] ||^2 ,   x >= 0.

The L-curve criterion (paper Eqs. 3-4, Fig. 5) then picks lambda at the corner
of the log(solution norm) vs log(misfit norm) trade-off curve.

No GUI dependency, so it can be unit-tested on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import nnls

from impedance_fit import ImpedanceData


# --------------------------------------------------------------------------- #
#  tau grid
# --------------------------------------------------------------------------- #
def make_tau_grid(freq: np.ndarray, ppd: int = 10, extend_decades: float = 1.0
                  ) -> np.ndarray:
    """Log-spaced relaxation-time grid.

    The characteristic relaxation time of an arc peaking at frequency f is
    tau = 1/(2 pi f), so the measured band f in [f_min, f_max] maps to
    tau in [1/(2 pi f_max), 1/(2 pi f_min)].  The grid is widened by
    `extend_decades` on each side (the tails of a peak sitting at the edge of
    the band still need support) and sampled at `ppd` points per decade.
    """
    f = np.asarray(freq, float)
    f = f[np.isfinite(f) & (f > 0)]
    tau_min = 1.0 / (2.0 * np.pi * f.max())
    tau_max = 1.0 / (2.0 * np.pi * f.min())
    lo = np.log10(tau_min) - extend_decades
    hi = np.log10(tau_max) + extend_decades
    n = max(int(round((hi - lo) * ppd)) + 1, 8)
    return np.logspace(lo, hi, n)


# --------------------------------------------------------------------------- #
#  Kernel and regularization matrices
# --------------------------------------------------------------------------- #
def build_kernel(freq: np.ndarray, tau: np.ndarray, fit_r_inf: bool = True
                 ) -> Tuple[np.ndarray, float]:
    """Real stacked kernel A (2N x (M[+1])) and the ln-tau spacing dlntau.

    Row block 1 = real part, row block 2 = imaginary part of

        A_col_m(w_n) = dlntau / (1 + j w_n tau_m).

    When `fit_r_inf` an extra unpenalised column (ones in the real block,
    zeros in the imaginary block) lets the ohmic offset R_inf be fitted too.
    """
    w = 2.0 * np.pi * np.asarray(freq, float)
    tau = np.asarray(tau, float)
    dlnt = float(np.mean(np.diff(np.log(tau))))          # uniform on log grid

    wt = np.outer(w, tau)                                # (N, M)
    denom = 1.0 + wt ** 2
    k_real = (1.0 / denom) * dlnt
    k_imag = (-wt / denom) * dlnt

    if fit_r_inf:
        k_real = np.hstack([k_real, np.ones((len(w), 1))])
        k_imag = np.hstack([k_imag, np.zeros((len(w), 1))])

    A = np.vstack([k_real, k_imag])
    return A, dlnt


def reg_matrix(m: int, order: int, has_r_inf: bool) -> np.ndarray:
    """Regularization (roughness) operator L acting on the g vector.

    order 0 -> identity  (solution norm = sqrt(sum g^2), i.e. paper Eq. 3),
    order 1 -> first difference, order 2 -> second difference.  The R_inf
    column (if present) is never penalised (its L column is zero)."""
    if order == 0:
        L = np.eye(m)
    elif order == 1:
        L = np.zeros((m - 1, m))
        for i in range(m - 1):
            L[i, i] = -1.0
            L[i, i + 1] = 1.0
    elif order == 2:
        L = np.zeros((m - 2, m))
        for i in range(m - 2):
            L[i, i] = 1.0
            L[i, i + 1] = -2.0
            L[i, i + 2] = 1.0
    else:
        raise ValueError("order must be 0, 1 or 2")
    if has_r_inf:
        L = np.hstack([L, np.zeros((L.shape[0], 1))])    # do not penalise R_inf
    return L


# --------------------------------------------------------------------------- #
#  Weighting
# --------------------------------------------------------------------------- #
def _row_weights(z: np.ndarray, weighting: str) -> np.ndarray:
    """Per-frequency weight applied to the real and imaginary rows alike."""
    weighting = weighting.lower()
    mod = np.abs(z)
    mod[mod == 0] = np.finfo(float).eps
    if weighting == "unit":
        return np.ones_like(mod)
    if weighting == "modulus":
        return 1.0 / mod
    raise ValueError(f"unknown weighting '{weighting}'")


# --------------------------------------------------------------------------- #
#  Single-lambda solve
# --------------------------------------------------------------------------- #
@dataclass
class DRTSolution:
    tau: np.ndarray            # relaxation times (s)
    gamma: np.ndarray          # g(ln tau) distribution, Ohm per unit ln tau
    freq_tau: np.ndarray       # 1/(2 pi tau) — frequency axis for the DRT
    r_inf: float               # fitted ohmic offset (Ohm)
    lam: float                 # regularization parameter (k_reg)
    dlnt: float                # ln-tau spacing
    z_model: np.ndarray        # reconstructed impedance at the data frequencies
    solution_norm: float       # eta = || L g ||   (paper Eq. 3 for order 0)
    misfit_norm: float         # rho = || W(Ax-b) || / sqrt(N)  (paper Eq. 4)
    r_pol: float               # total polarization resistance = area of g

    @property
    def order_freq(self) -> np.ndarray:
        return self.freq_tau


def solve_drt(
    data: ImpedanceData,
    tau: np.ndarray,
    lam: float,
    order: int = 1,
    weighting: str = "modulus",
    fit_r_inf: bool = True,
) -> DRTSolution:
    """Solve the regularized non-negative DRT for one lambda (k_reg)."""
    freq = np.asarray(data.freq, float)
    z = data.z
    A, dlnt = build_kernel(freq, tau, fit_r_inf=fit_r_inf)
    m_tau = len(tau)
    L = reg_matrix(m_tau, order, has_r_inf=fit_r_inf)

    # row weighting (real & imag rows share the per-frequency weight)
    wrow = _row_weights(z, weighting)
    W = np.concatenate([wrow, wrow])                     # (2N,)
    b = np.concatenate([z.real, z.imag])

    Aw = A * W[:, None]
    bw = b * W

    # augmented non-negative least squares:  [Aw ; lam L] x = [bw ; 0]
    Aa = np.vstack([Aw, lam * L])
    ba = np.concatenate([bw, np.zeros(L.shape[0])])
    x, _ = nnls(Aa, ba, maxiter=20 * Aa.shape[1])

    g = x[:m_tau]
    r_inf = float(x[m_tau]) if fit_r_inf else 0.0

    z_model = A @ x                                      # stacked real/imag
    z_model = z_model[:len(freq)] + 1j * z_model[len(freq):]

    resid_w = Aw @ x - bw
    misfit = float(np.linalg.norm(resid_w) / np.sqrt(len(freq)))
    sol_norm = float(np.linalg.norm(L @ x))
    r_pol = float(np.sum(g) * dlnt)

    return DRTSolution(
        tau=np.asarray(tau, float),
        gamma=g,
        freq_tau=1.0 / (2.0 * np.pi * np.asarray(tau, float)),
        r_inf=r_inf,
        lam=float(lam),
        dlnt=dlnt,
        z_model=z_model,
        solution_norm=sol_norm,
        misfit_norm=misfit,
        r_pol=r_pol,
    )


# --------------------------------------------------------------------------- #
#  L-curve
# --------------------------------------------------------------------------- #
@dataclass
class LCurve:
    lambdas: np.ndarray        # k_reg values scanned (ascending)
    rho: np.ndarray            # misfit norm  (x-axis)
    eta: np.ndarray            # solution norm (y-axis)
    curvature: np.ndarray      # curvature of the log-log curve
    i_opt: int                 # index of the maximum-curvature corner
    lam_opt: float             # optimum k_reg
    solutions: List[DRTSolution] = field(default_factory=list)


def default_lambda_grid(data: ImpedanceData, tau: np.ndarray,
                        weighting: str = "modulus", n: int = 60,
                        fit_r_inf: bool = True) -> np.ndarray:
    """A broad, data-scaled log grid of k_reg values for the L-curve.

    The useful range of lambda scales with the singular values of the
    (weighted) kernel, so the grid is centred on the largest singular value
    and spans ~7 decades below it — wide enough to contain both the
    over-smoothed knee and the under-smoothed vertical branch of the L curve."""
    A, _ = build_kernel(data.freq, tau, fit_r_inf=fit_r_inf)
    wrow = _row_weights(data.z, weighting)
    W = np.concatenate([wrow, wrow])
    s1 = np.linalg.norm(A * W[:, None], 2)               # largest singular value
    hi = np.log10(s1) + 0.5
    lo = hi - 7.5
    return np.logspace(lo, hi, n)


def _lcurve_curvature(rho: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """Curvature of the (log rho, log eta) curve via finite differences.

    Positive curvature = concave toward the origin (the L-corner). Endpoints
    get zero curvature so they are never chosen as the corner."""
    x = np.log10(np.maximum(rho, 1e-300))
    y = np.log10(np.maximum(eta, 1e-300))
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denom = (dx * dx + dy * dy) ** 1.5
    denom[denom == 0] = np.finfo(float).eps
    kappa = (dx * ddy - dy * ddx) / denom
    kappa[0] = kappa[-1] = 0.0
    return kappa


def _pick_corner(kappa: np.ndarray, eta: np.ndarray, frac: float = 0.5) -> int:
    """Choose the L-curve corner from the curvature profile.

    A regularized L-curve can show two positive-curvature bumps: the genuine
    elbow at the under/over-smoothing transition, and a spurious one far out in
    the over-smoothed tail (where the solution has already collapsed). We take
    every interior local maximum of the curvature that reaches at least `frac`
    of the largest curvature, then keep the one with the SMALLEST lambda
    (leftmost / least-smoothed) — that is the true corner. Falls back to the
    global maximum if no interior peak qualifies."""
    n = len(kappa)
    valid = eta > (eta.max() * 1e-6)
    k = np.where(valid, kappa, -np.inf)
    kmax = np.max(kappa[1:-1]) if n > 2 else np.max(kappa)
    if kmax <= 0:
        return int(np.argmax(k))
    thr = frac * kmax
    maxima = [i for i in range(1, n - 1)
              if k[i] >= thr and k[i] >= k[i - 1] and k[i] >= k[i + 1]]
    if maxima:
        return maxima[0]                                 # smallest lambda (grid ascending)
    return int(np.argmax(k))


def compute_lcurve(
    data: ImpedanceData,
    tau: np.ndarray,
    lambdas: Optional[np.ndarray] = None,
    order: int = 1,
    weighting: str = "modulus",
    fit_r_inf: bool = True,
) -> LCurve:
    """Scan lambda, build the L-curve and locate its maximum-curvature corner."""
    if lambdas is None:
        lambdas = default_lambda_grid(data, tau, weighting, fit_r_inf=fit_r_inf)
    lambdas = np.sort(np.asarray(lambdas, float))

    sols, rho, eta = [], [], []
    for lam in lambdas:
        s = solve_drt(data, tau, lam, order=order, weighting=weighting,
                      fit_r_inf=fit_r_inf)
        sols.append(s)
        rho.append(s.misfit_norm)
        eta.append(s.solution_norm)
    rho = np.array(rho)
    eta = np.array(eta)

    kappa = _lcurve_curvature(rho, eta)
    i_opt = _pick_corner(kappa, eta)

    return LCurve(
        lambdas=lambdas, rho=rho, eta=eta, curvature=kappa,
        i_opt=i_opt, lam_opt=float(lambdas[i_opt]), solutions=sols,
    )


# --------------------------------------------------------------------------- #
#  Peak analysis
# --------------------------------------------------------------------------- #
@dataclass
class DRTPeak:
    tau: float                 # peak relaxation time (s)
    freq: float                # peak frequency 1/(2 pi tau) (Hz)
    gamma: float               # peak height (Ohm per unit ln tau)
    resistance: float          # integrated resistance of the peak (Ohm)
    capacitance: float         # C = tau / R (F)


def find_peaks(sol: DRTSolution, rel_height: float = 0.02) -> List[DRTPeak]:
    """Local maxima of g(ln tau); each peak's resistance is the trapezoidal
    integral of g over the valley-to-valley interval around it, and its
    capacitance follows from C = tau_peak / R."""
    g = sol.gamma
    tau = sol.tau
    lnt = np.log(tau)
    if len(g) < 3 or g.max() <= 0:
        return []
    thr = rel_height * g.max()

    # interior local maxima above the threshold
    peak_idx = [i for i in range(1, len(g) - 1)
                if g[i] > g[i - 1] and g[i] >= g[i + 1] and g[i] > thr]

    # valley (local minima) positions bounding each peak's integration window
    valleys = [0] + [i for i in range(1, len(g) - 1)
                     if g[i] <= g[i - 1] and g[i] < g[i + 1]] + [len(g) - 1]

    peaks: List[DRTPeak] = []
    for pi in peak_idx:
        left = max([v for v in valleys if v <= pi], default=0)
        right = min([v for v in valleys if v >= pi], default=len(g) - 1)
        R = float(np.trapezoid(g[left:right + 1], lnt[left:right + 1]))
        tp = float(tau[pi])
        C = tp / R if R > 0 else np.nan
        peaks.append(DRTPeak(tau=tp, freq=1.0 / (2.0 * np.pi * tp),
                             gamma=float(g[pi]), resistance=R, capacitance=C))
    peaks.sort(key=lambda p: p.freq, reverse=True)       # high-f first
    return peaks
