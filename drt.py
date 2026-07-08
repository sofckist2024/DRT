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
def n_extra_cols(fit_r_inf: bool, fit_ind: bool) -> int:
    """Number of non-distribution columns: R_inf (1) + inductance (2, ±)."""
    return (1 if fit_r_inf else 0) + (2 if fit_ind else 0)


def build_kernel(freq: np.ndarray, tau: np.ndarray, fit_r_inf: bool = True,
                 fit_ind: bool = True) -> Tuple[np.ndarray, float]:
    """Real stacked kernel A (2N x (M + extra)) and the ln-tau spacing dlntau.

    Row block 1 = real part, row block 2 = imaginary part of

        A_col_m(w_n) = dlntau / (1 + j w_n tau_m).

    Extra unpenalised columns (appended after the M distribution columns):

    * `fit_r_inf`: the ohmic offset R_inf  (ones in real block, zeros in imag).
    * `fit_ind`:   a series inductance  Z = j w L, split into TWO non-negative
      columns (+w and -w in the imag block) so the net L = x_+ - x_- can take
      either sign. This lets the DRT absorb any residual inductance left in the
      spectrum — essential, because the pure DRT kernel 1/(1+jwt) can only
      produce capacitive (Z'' < 0) response and cannot represent an inductive
      tail. Without it, leftover inductive points inflate the misfit floor and
      the L-curve corner collapses to a hugely over-smoothed k_reg.
    """
    w = 2.0 * np.pi * np.asarray(freq, float)
    tau = np.asarray(tau, float)
    dlnt = float(np.mean(np.diff(np.log(tau))))          # uniform on log grid

    wt = np.outer(w, tau)                                # (N, M)
    denom = 1.0 + wt ** 2
    k_real = (1.0 / denom) * dlnt
    k_imag = (-wt / denom) * dlnt

    extra_real, extra_imag = [], []
    if fit_r_inf:
        extra_real.append(np.ones((len(w), 1)))
        extra_imag.append(np.zeros((len(w), 1)))
    if fit_ind:
        extra_real.append(np.zeros((len(w), 2)))
        extra_imag.append(np.column_stack([w, -w]))      # +wL and -wL branches
    if extra_real:
        k_real = np.hstack([k_real] + extra_real)
        k_imag = np.hstack([k_imag] + extra_imag)

    A = np.vstack([k_real, k_imag])
    return A, dlnt


def reg_matrix(m: int, order: int, n_extra: int) -> np.ndarray:
    """Regularization (roughness) operator L acting on the g vector.

    order 0 -> identity  (solution norm = sqrt(sum g^2), i.e. paper Eq. 3),
    order 1 -> first difference, order 2 -> second difference.  The `n_extra`
    non-distribution columns (R_inf, inductance) are never penalised (their L
    columns are zero)."""
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
    if n_extra:
        L = np.hstack([L, np.zeros((L.shape[0], n_extra))])
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
    l_ind: float               # fitted residual series inductance (H)
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
    weighting: str = "unit",
    fit_r_inf: bool = True,
    fit_ind: bool = True,
) -> DRTSolution:
    """Solve the regularized non-negative DRT for one lambda (k_reg)."""
    freq = np.asarray(data.freq, float)
    z = data.z
    A, dlnt = build_kernel(freq, tau, fit_r_inf=fit_r_inf, fit_ind=fit_ind)
    m_tau = len(tau)
    n_extra = n_extra_cols(fit_r_inf, fit_ind)
    L = reg_matrix(m_tau, order, n_extra)

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
    idx = m_tau
    r_inf = float(x[idx]) if fit_r_inf else 0.0
    idx += 1 if fit_r_inf else 0
    # net inductance from the +/- branch pair (angular-freq column was w, so the
    # coefficient difference IS L in henries)
    l_ind = float(x[idx] - x[idx + 1]) if fit_ind else 0.0

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
        l_ind=l_ind,
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
                        weighting: str = "unit", n: int = 60,
                        fit_r_inf: bool = True, fit_ind: bool = True) -> np.ndarray:
    """A broad, data-scaled log grid of k_reg values for the L-curve.

    The regularization lambda*L acts ONLY on the M distribution columns, so the
    useful lambda range scales with the singular values of the weighted
    *distribution* submatrix — NOT the full kernel. The R_inf / inductance
    columns are unpenalised and can have enormous magnitude (the inductance
    column values are ~w, up to ~1e7), which would otherwise blow the grid up
    into the fully over-smoothed regime. The grid is centred on that submatrix's
    largest singular value and spans ~8 decades below it, covering both the
    over-smoothed knee and the under-smoothed branch of the L curve."""
    A, _ = build_kernel(data.freq, tau, fit_r_inf=fit_r_inf, fit_ind=fit_ind)
    wrow = _row_weights(data.z, weighting)
    W = np.concatenate([wrow, wrow])
    A_dist = (A * W[:, None])[:, :len(tau)]              # distribution columns only
    s1 = np.linalg.norm(A_dist, 2)                       # its largest singular value
    hi = np.log10(s1) + 0.5
    lo = hi - 8.0
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


def _pick_knee(rho: np.ndarray, tol: float = 0.05) -> int:
    """Choose the operating lambda at the 'knee' of the misfit vs lambda curve.

    For DRT the misfit norm rho stays essentially flat at its noise floor across
    a wide range of small lambda (the fit is insensitive to regularization until
    lambda gets large), then rises as over-smoothing sets in. The pure L-curve
    max-curvature corner is unreliable on this flat-misfit shape — its sharpest
    turn sits deep in the over-smoothed branch, merging real peaks into one.

    Instead we take the knee: the LARGEST lambda whose misfit is still within
    `tol` (fractional) of the minimum misfit. This applies as much smoothing as
    possible WITHOUT degrading the fit beyond the noise floor — i.e. it removes
    spurious oscillations for free but stops before it starts hiding real
    structure. `lambdas` must be ascending."""
    rmin = float(np.min(rho))
    thr = rmin * (1.0 + tol)
    ok = np.where(rho <= thr)[0]
    return int(ok.max()) if len(ok) else int(np.argmin(rho))


def compute_lcurve(
    data: ImpedanceData,
    tau: np.ndarray,
    lambdas: Optional[np.ndarray] = None,
    order: int = 1,
    weighting: str = "unit",
    fit_r_inf: bool = True,
    fit_ind: bool = True,
    knee_tol: float = 0.05,
) -> LCurve:
    """Scan lambda, build the L-curve and locate its corner (misfit knee)."""
    if lambdas is None:
        lambdas = default_lambda_grid(data, tau, weighting, fit_r_inf=fit_r_inf,
                                      fit_ind=fit_ind)
    lambdas = np.sort(np.asarray(lambdas, float))

    sols, rho, eta = [], [], []
    for lam in lambdas:
        s = solve_drt(data, tau, lam, order=order, weighting=weighting,
                      fit_r_inf=fit_r_inf, fit_ind=fit_ind)
        sols.append(s)
        rho.append(s.misfit_norm)
        eta.append(s.solution_norm)
    rho = np.array(rho)
    eta = np.array(eta)

    kappa = _lcurve_curvature(rho, eta)
    i_opt = _pick_knee(rho, tol=knee_tol)

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


def find_peaks(sol: DRTSolution, rel_height: float = 0.05,
               min_area_frac: float = 0.05) -> List[DRTPeak]:
    """Local maxima of g(ln tau); each peak's resistance is the trapezoidal
    integral of g over the valley-to-valley interval around it, and its
    capacitance follows from C = tau_peak / R.

    `rel_height` drops maxima shorter than this fraction of the tallest peak;
    `min_area_frac` then drops peaks whose integrated resistance is below this
    fraction of the total — this suppresses the small ripples that a lightly
    regularized (under-smoothed) solution sprinkles between the real arcs, so
    the reported peak list reflects genuine processes, not numerical wiggles."""
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

    # drop numerical ripples: keep only peaks carrying a meaningful share of R
    total_R = sum(p.resistance for p in peaks)
    if total_R > 0 and min_area_frac > 0:
        peaks = [p for p in peaks if p.resistance >= min_area_frac * total_R]

    peaks.sort(key=lambda p: p.freq, reverse=True)       # high-f first
    return peaks
