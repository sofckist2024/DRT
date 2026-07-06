"""
Impedance (EIS) fitting core.

Equivalent circuit:  L - Rs - (R1-CPE1) - (R2-CPE2) - ... - (Rn-CPEn)

    Z(w) = jwL + Rs + sum_i  R_i / (1 + R_i * Q_i * (jw)^n_i)

where each (R_i - CPE_i) is a resistor in parallel with a constant phase element,
CPE impedance  Z_CPE = 1 / (Q (jw)^n).

This module has no GUI dependency so it can be unit-tested on its own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares


# --------------------------------------------------------------------------- #
#  Data container
# --------------------------------------------------------------------------- #
@dataclass
class ImpedanceData:
    freq: np.ndarray          # Hz
    z_real: np.ndarray        # Ohm
    z_imag: np.ndarray        # Ohm  (sign as stored in file, i.e. Z'')

    @property
    def omega(self) -> np.ndarray:
        return 2.0 * np.pi * self.freq

    @property
    def z(self) -> np.ndarray:
        return self.z_real + 1j * self.z_imag

    def __len__(self) -> int:
        return len(self.freq)


# --------------------------------------------------------------------------- #
#  ZView / ZPlot .z file parser
# --------------------------------------------------------------------------- #
def _is_number(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def _split(line: str) -> List[str]:
    # ZView .z files are comma separated, but tolerate tabs / whitespace too.
    if "," in line:
        parts = [p.strip() for p in line.split(",")]
    else:
        parts = line.split()
    return [p for p in parts if p != ""]


_HEADER_END_MARKERS = ("end comments", "end header")


def _data_start(lines: List[str]) -> int:
    """Index of the first data row."""
    for i, ln in enumerate(lines):
        low = ln.lower()
        if any(m in low for m in _HEADER_END_MARKERS):
            return i + 1
    # no marker: first line that looks like >=3 numbers starts the block
    for i, ln in enumerate(lines):
        toks = _split(ln)
        if len(toks) >= 3 and sum(_is_number(t) for t in toks) >= 3 \
                and all(_is_number(t) for t in toks):
            return i
    return 0


def detect_columns(text: str) -> Optional[Tuple[int, int, int]]:
    """Identify (freq, Z', Z'') column indices from a *named* header line such as

        Freq(Hz)  Ampl  Bias  Time(Sec)  Z'(a)  Z''(b)  GD  Error  Range

    (used by ZView / MultiStat AC-data exports).  Returns None if no such
    header line is present.  Z'' (double prime) is matched before Z' so the
    imaginary column is not mistaken for the real one."""
    for ln in text.splitlines():
        toks = _split(ln)
        if len(toks) < 3 or all(_is_number(t) for t in toks):
            continue
        low = [t.lower() for t in toks]
        if not any("freq" in t for t in low):
            continue
        cf = next((i for i, t in enumerate(low) if "freq" in t), None)
        ci = next((i for i, t in enumerate(low)
                   if ("z" in t and ("''" in t or '"' in t)) or "zimag" in t
                   or t in ("zi", "-z''", "z2")), None)
        cr = next((i for i, t in enumerate(low)
                   if ("z'" in t and "''" not in t and '"' not in t)
                   or "zreal" in t or t in ("zr", "z1")), None)
        if None not in (cf, cr, ci) and len({cf, cr, ci}) == 3:
            return cf, cr, ci
    return None


def _auto_freq_col(arr: np.ndarray) -> Optional[int]:
    """Column that spans the most decades and is strictly positive -> frequency."""
    best, best_span = None, -1.0
    for j in range(arr.shape[1]):
        col = arr[:, j]
        if np.all(np.isfinite(col)) and np.all(col > 0) and np.std(col) > 0:
            span = float(np.log10(col.max()) - np.log10(col.min()))
            if span > best_span:
                best_span, best = span, j
    return best


def suggest_columns(text: str) -> Tuple[int, int, int]:
    """Best-guess (freq, Z', Z'') column indices. Never fails: uses the named
    header if present, otherwise a positional default, and self-corrects if the
    chosen frequency column turns out to be constant (the classic mis-map)."""
    rows, ncol = preview_columns(text, max_rows=400)
    named = detect_columns(text)
    if named is not None:
        cf, cr, ci = named
    elif ncol >= 6:
        cf, cr, ci = 0, 4, 5
    elif ncol >= 3:
        cf, cr, ci = 0, 1, 2
    else:
        cf, cr, ci = 0, min(1, ncol - 1), min(2, ncol - 1)

    try:
        arr = np.array([[float(x) for x in r[:ncol]] for r in rows], dtype=float)
        if arr.shape[0] >= 2:
            col = arr[:, cf]
            if np.std(col) == 0 or not np.all(col > 0):   # freq must vary & be >0
                cand = _auto_freq_col(arr)
                if cand is not None:
                    cf = cand
    except Exception:  # noqa: BLE001
        pass
    return cf, cr, ci


def parse_z_file(
    text: str,
    col_freq: Optional[int] = None,
    col_zreal: Optional[int] = None,
    col_zimag: Optional[int] = None,
) -> ImpedanceData:
    """Parse the text of a ZView / MultiStat .z file (or a generic text table).

    The numeric block follows an "End Header:"/"End Comments" marker (or the
    first all-numeric line).  Columns are auto-detected from a named header line
    when available (e.g. ``Freq(Hz) ... Z'(a) Z''(b) ...``), otherwise guessed
    by position/content.  Any of the three indices can be overridden.
    """
    lines = text.splitlines()
    start = _data_start(lines)

    rows: List[List[float]] = []
    for ln in lines[start:]:
        toks = _split(ln)
        if len(toks) < 3:
            continue
        if not all(_is_number(t) for t in toks):
            continue
        rows.append([float(t) for t in toks])

    if not rows:
        raise ValueError("파일에서 숫자 데이터 행을 찾지 못했습니다. 형식을 확인하세요.")

    ncol = min(len(r) for r in rows)
    arr = np.array([r[:ncol] for r in rows], dtype=float)

    # decide columns ---------------------------------------------------------
    if col_freq is None or col_zreal is None or col_zimag is None:
        cf, cr, ci = suggest_columns(text)
        col_freq = cf if col_freq is None else col_freq
        col_zreal = cr if col_zreal is None else col_zreal
        col_zimag = ci if col_zimag is None else col_zimag

    for c in (col_freq, col_zreal, col_zimag):
        if c >= ncol:
            raise ValueError(f"열 인덱스 {c} 가 데이터 열 수({ncol})를 벗어났습니다.")

    data = ImpedanceData(
        freq=arr[:, col_freq],
        z_real=arr[:, col_zreal],
        z_imag=arr[:, col_zimag],
    )

    # keep only positive, finite frequencies, sort high -> low
    m = np.isfinite(data.freq) & (data.freq > 0)
    m &= np.isfinite(data.z_real) & np.isfinite(data.z_imag)
    order = np.argsort(data.freq[m])[::-1]
    idx = np.where(m)[0][order]
    return ImpedanceData(data.freq[idx], data.z_real[idx], data.z_imag[idx])


def preview_columns(text: str, max_rows: int = 5) -> Tuple[List[List[str]], int]:
    """Return the first few numeric data rows (as strings) and the column count,
    so the UI can show the user which column is which."""
    lines = text.splitlines()
    start = _data_start(lines)
    rows = []
    for ln in lines[start:]:
        toks = _split(ln)
        if len(toks) >= 3 and all(_is_number(t) for t in toks):
            rows.append(toks)
        if len(rows) >= max_rows:
            break
    ncol = min((len(r) for r in rows), default=0)
    return rows, ncol


# --------------------------------------------------------------------------- #
#  High-frequency artifact detection
# --------------------------------------------------------------------------- #
def detect_hf_artifact(
    data: ImpedanceData,
    tol_frac: float = 0.08,
    min_inductive: int = 3,
    min_keep: int = 5,
) -> np.ndarray:
    """Boolean keep-mask that drops the high-frequency 'twist' at the top of the
    inductive tail.

    Physics: at high frequency the (R-CPE) arcs collapse and Z → Rs + jwL, so
    the inductive branch (Z'' > 0) should be a straight *vertical* line at
    Z' ≈ Rs. Real data often curls / loops at the very highest frequencies
    (mutual-inductance and cable artifacts) instead of staying vertical. This
    walks down from the highest frequency and flags the contiguous top block
    whose Z' departs from that vertical line by more than `tol_frac` of the arc
    width, so only the clean vertical part (and the arcs below) are fitted.

    Returns an all-True mask (exclude nothing) when there is no clear inductive
    tail, when the tail is already vertical, or when exclusion would leave fewer
    than `min_keep` points."""
    zr = np.asarray(data.z_real, float)
    zi = np.asarray(data.z_imag, float)
    f = np.asarray(data.freq, float)
    keep = np.ones(len(f), dtype=bool)

    ind = np.where(zi > 0)[0]                        # inductive points (Z'' > 0)
    if len(ind) < min_inductive:
        return keep

    order_up = ind[np.argsort(f[ind])]              # inductive idx, low → high f
    n_base = max(3, len(ind) // 5)
    Rs_base = float(np.median(zr[order_up[:n_base]]))    # foot of the vertical line
    Rp_scale = max(float(np.max(zr) - Rs_base), float(np.ptp(zr)), 1e-9)
    tol = max(tol_frac * Rp_scale, 1e-12)

    # from the highest frequency downward, drop the contiguous run whose Z'
    # departs from the vertical line; stop at the first in-band (vertical) point
    cut_freq = None
    for k in order_up[::-1]:                         # high → low f
        if abs(zr[k] - Rs_base) > tol:
            cut_freq = f[k]
        else:
            break
    if cut_freq is not None:
        keep[f >= cut_freq] = False

    if int(keep.sum()) < min_keep:                  # safety: never over-trim
        keep[:] = True
    return keep


def subset(data: ImpedanceData, mask: np.ndarray) -> ImpedanceData:
    """Return a new ImpedanceData containing only the points where mask is True."""
    mask = np.asarray(mask, dtype=bool)
    return ImpedanceData(
        freq=data.freq[mask], z_real=data.z_real[mask], z_imag=data.z_imag[mask],
    )


# --------------------------------------------------------------------------- #
#  Circuit model
# --------------------------------------------------------------------------- #
def circuit_impedance(params: np.ndarray, freq: np.ndarray, n_elem: int) -> np.ndarray:
    """params = [L, Rs, R1, Q1, n1, R2, Q2, n2, ...]"""
    w = 2.0 * np.pi * freq
    jw = 1j * w
    L = params[0]
    Rs = params[1]
    z = 1j * w * L + Rs
    for k in range(n_elem):
        R = params[2 + 3 * k]
        Q = params[3 + 3 * k]
        n = params[4 + 3 * k]
        z = z + R / (1.0 + R * Q * np.power(jw, n))
    return z


def param_labels(n_elem: int) -> List[str]:
    labels = ["L (H)", "Rs (Ohm)"]
    for k in range(1, n_elem + 1):
        labels += [f"R{k} (Ohm)", f"Q{k} (S·s^n)", f"n{k}"]
    return labels


# --------------------------------------------------------------------------- #
#  Weighting
# --------------------------------------------------------------------------- #
def _weights(z: np.ndarray, kind: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (w_real, w_imag) weight arrays for the residuals."""
    kind = kind.lower()
    mod = np.abs(z)
    mod[mod == 0] = np.finfo(float).eps
    if kind == "unit":
        w = np.ones_like(mod)
        return w, w
    if kind == "modulus":
        w = 1.0 / mod
        return w, w
    if kind == "proportional":
        wr = 1.0 / np.maximum(np.abs(z.real), np.finfo(float).eps)
        wi = 1.0 / np.maximum(np.abs(z.imag), np.finfo(float).eps)
        return wr, wi
    raise ValueError(f"unknown weighting '{kind}'")


# --------------------------------------------------------------------------- #
#  Initial guess
# --------------------------------------------------------------------------- #
def initial_guess(data: ImpedanceData, n_elem: int,
                  fc: Optional[List[float]] = None,
                  n0: float = 0.85,
                  r_frac: Optional[List[float]] = None) -> np.ndarray:
    """Build a parameter vector from data-derived scale estimates.

    `fc` = characteristic frequency of each element (Hz). If None they are
    spread log-uniformly over the measured range. `r_frac` = how the total
    polarization resistance is split between elements (defaults to equal)."""
    f = data.freq
    zr = data.z_real
    zi = data.z_imag
    w = 2.0 * np.pi * f

    Rs0 = max(float(np.min(zr)), 1e-6)

    # inductance from the most inductive (Zimag > 0) high-freq point
    L0 = 1e-7
    hi = np.argmax(f)
    if zi[hi] > 0 and w[hi] > 0:
        L0 = max(zi[hi] / w[hi], 1e-9)

    Rp = max(float(np.max(zr)) - Rs0, abs(float(np.max(zr))) * 0.1, 1e-3)
    if r_frac is None:
        r_frac = [1.0 / n_elem] * n_elem

    fmin, fmax = float(np.min(f)), float(np.max(f))
    if fc is None:
        lo, hi_ = np.log10(fmin), np.log10(fmax)
        span = hi_ - lo
        lo += 0.1 * span
        hi_ -= 0.1 * span
        if n_elem == 1:
            fc = [10 ** ((lo + hi_) / 2)]
        else:
            fc = list(10 ** np.linspace(hi_, lo, n_elem))  # high-f element first

    params = [L0, Rs0]
    for i in range(n_elem):
        Ri = max(Rp * r_frac[i], 1e-3)
        wc = 2.0 * np.pi * fc[i]
        Q0 = 1.0 / (Ri * (wc ** n0))
        params += [Ri, Q0, n0]
    return np.array(params, dtype=float)


def _candidate_starts(data: ImpedanceData, n_elem: int, n_starts: int) -> List[np.ndarray]:
    """A deterministic + seeded-random set of starting guesses that place the
    element characteristic frequencies across many different sub-ranges of the
    measured spectrum. Multi-start is what makes the fit robust to local minima."""
    fmin, fmax = float(np.min(data.freq)), float(np.max(data.freq))
    lo, hi = np.log10(fmin), np.log10(fmax)
    starts: List[np.ndarray] = []

    # 1) evenly spread over the full range, a few values of n
    for n0 in (0.9, 0.75, 1.0):
        starts.append(initial_guess(data, n_elem, n0=n0))

    # 2) clustered in the low / mid / high thirds (arcs are often bunched)
    thirds = [(lo, lo + (hi - lo) / 3),
              (lo + (hi - lo) / 3, lo + 2 * (hi - lo) / 3),
              (lo + 2 * (hi - lo) / 3, hi)]
    for a, b in thirds:
        if n_elem == 1:
            fc = [10 ** ((a + b) / 2)]
        else:
            fc = list(10 ** np.linspace(b, a, n_elem))
        starts.append(initial_guess(data, n_elem, fc=fc, n0=0.85))

    # 3) seeded random placements (no wall-clock randomness -> reproducible)
    rng = np.random.default_rng(12345)
    while len(starts) < n_starts:
        cf = np.sort(rng.uniform(lo, hi, n_elem))[::-1]
        fc = list(10 ** cf)
        n0 = float(rng.uniform(0.6, 1.0))
        frac = rng.uniform(0.2, 1.0, n_elem)
        frac = list(frac / frac.sum())
        starts.append(initial_guess(data, n_elem, fc=fc, n0=n0, r_frac=frac))

    return starts


def default_bounds(n_elem: int) -> Tuple[np.ndarray, np.ndarray]:
    lo = [0.0, 0.0]
    hi = [np.inf, np.inf]
    for _ in range(n_elem):
        lo += [0.0, 0.0, 0.0]      # R, Q, n
        hi += [np.inf, np.inf, 1.0]
    return np.array(lo), np.array(hi)


# --------------------------------------------------------------------------- #
#  Fit
# --------------------------------------------------------------------------- #
@dataclass
class FitResult:
    params: np.ndarray
    perror: np.ndarray
    labels: List[str]
    n_elem: int
    weighting: str
    chi_square: float          # sum of weighted squared residuals
    chi_square_reduced: float
    success: bool
    message: str
    z_fit: np.ndarray = field(default=None)

    def as_rows(self) -> List[Tuple[str, float, float]]:
        return list(zip(self.labels, self.params, self.perror))


def fit_impedance(
    data: ImpedanceData,
    n_elem: int,
    weighting: str = "modulus",
    p0: Optional[np.ndarray] = None,
    max_nfev: int = 20000,
    n_starts: int = 24,
    fix_L: bool = False,
) -> FitResult:
    """Fit the equivalent circuit. Runs a multi-start least-squares to avoid
    local minima; if `p0` is given it is used as the (single) start.

    `fix_L=True` pins the inductance L to ~0 (used when fitting data whose
    inductive part has already been removed)."""
    lo, hi = default_bounds(n_elem)
    if fix_L:
        hi = hi.copy()
        hi[0] = 1e-30              # L ~ 0 (least_squares needs lo < hi strictly)
    hi_clip = np.where(np.isinf(hi), 1e30, hi)
    wr, wi = _weights(data.z, weighting)

    def residuals(p: np.ndarray) -> np.ndarray:
        zm = circuit_impedance(p, data.freq, n_elem)
        return np.concatenate([(data.z_real - zm.real) * wr,
                               (data.z_imag - zm.imag) * wi])

    if p0 is not None:
        starts = [np.asarray(p0, dtype=float)]
    else:
        starts = _candidate_starts(data, n_elem, n_starts)

    best = None
    best_cost = np.inf
    for start in starts:
        s = np.clip(start, lo + 1e-30, hi_clip)
        try:
            res = least_squares(
                residuals, s, bounds=(lo, hi),
                method="trf", x_scale="jac", max_nfev=max_nfev,
                ftol=1e-12, xtol=1e-12, gtol=1e-12,
            )
        except Exception:  # noqa: BLE001  (a bad start should not kill the fit)
            continue
        if res.cost < best_cost:
            best_cost = res.cost
            best = res

    if best is None:
        raise RuntimeError("모든 초기값에서 피팅에 실패했습니다.")
    res = best

    # parameter uncertainty from the Jacobian
    m2 = len(res.fun)
    dof = max(m2 - len(res.x), 1)
    chi2 = float(np.sum(res.fun ** 2))
    chi2_red = chi2 / dof
    perror = np.full_like(res.x, np.nan)
    try:
        J = res.jac
        # A parameter pinned at an active bound (e.g. L when fix_L=True) gets a
        # zero Jacobian column, which makes J.T@J singular. Estimate the
        # covariance from the sensitive (free) columns only and report 0 error
        # for the pinned ones, so the remaining parameters still get error bars.
        col_norm = np.linalg.norm(J, axis=0)
        free = col_norm > 1e-12 * max(col_norm.max(), 1e-30)
        cov = np.linalg.inv(J[:, free].T @ J[:, free]) * chi2_red
        perror = np.zeros_like(res.x)
        perror[free] = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
    except np.linalg.LinAlgError:
        pass

    return FitResult(
        params=res.x,
        perror=perror,
        labels=param_labels(n_elem),
        n_elem=n_elem,
        weighting=weighting,
        chi_square=chi2,
        chi_square_reduced=chi2_red,
        success=bool(res.success),
        message=str(res.message),
        z_fit=circuit_impedance(res.x, data.freq, n_elem),
    )


# --------------------------------------------------------------------------- #
#  Inductance correction
# --------------------------------------------------------------------------- #
def inductance_corrected(data: ImpedanceData, L: float) -> ImpedanceData:
    """Return a copy of `data` with the pure-inductive part jwL removed:

        Z_corr(w) = Z(w) - jwL      (Z' unchanged,  Z'' -> Z'' - wL)

    Wiring/lead inductance shows up as jwL and only distorts the imaginary
    part at high frequency (the low-freq tail dipping below the real axis in a
    Nyquist plot). Subtracting it leaves the ohmic + polarization response."""
    w = data.omega
    return ImpedanceData(
        freq=data.freq.copy(),
        z_real=data.z_real.copy(),
        z_imag=data.z_imag - w * L,
    )


@dataclass
class CorrectionResult:
    """Outcome of removing the fitted inductance and re-fitting."""
    L: float                     # inductance removed (H), from the original fit
    data_corr: ImpedanceData     # L-corrected measured data
    fit: FitResult               # re-fit of the corrected data (L pinned to ~0)
    Rs: float                    # ohmic resistance (Ohm)
    Rp_list: List[float]         # polarization resistance of each element (Ohm)
    Rp_total: float              # total polarization resistance = sum(Rp_list)


def remove_inductance(
    data: ImpedanceData,
    fit_result: FitResult,
    weighting: str = "modulus",
) -> CorrectionResult:
    """Report the inductance-free response of an existing fit.

    A series inductance contributes only jwL to the imaginary part and is
    mathematically separable from Rs and the (R-CPE) arcs, so the Rs / R_i that
    the fit already found ARE the inductance-free values. We therefore just zero
    L and rebuild the circuit response — we do NOT re-fit the corrected data,
    because a fresh fit can converge to a different local minimum and no longer
    match the original fit's arcs. The Rs / R_i (and their errors) are carried
    over from `fit_result` unchanged.

    `weighting` is accepted for backward compatibility and is no longer used.
    """
    L = float(fit_result.params[0])
    data_corr = inductance_corrected(data, L)

    params0 = np.array(fit_result.params, dtype=float).copy()
    params0[0] = 0.0                           # drop the series inductance
    perror0 = np.array(fit_result.perror, dtype=float).copy()
    if perror0.size:
        perror0[0] = 0.0
    model = FitResult(
        params=params0,
        perror=perror0,
        labels=list(fit_result.labels),
        n_elem=fit_result.n_elem,
        weighting=fit_result.weighting,
        chi_square=fit_result.chi_square,
        chi_square_reduced=fit_result.chi_square_reduced,
        success=fit_result.success,
        message=fit_result.message,
        z_fit=circuit_impedance(params0, data_corr.freq, fit_result.n_elem),
    )
    Rs = float(params0[1])
    Rp_list = [float(params0[2 + 3 * k]) for k in range(fit_result.n_elem)]
    return CorrectionResult(
        L=L,
        data_corr=data_corr,
        fit=model,
        Rs=Rs,
        Rp_list=Rp_list,
        Rp_total=float(sum(Rp_list)),
    )
