"""Self test for the DRT pipeline.

Builds a synthetic L - Rs - (R1-CPE1) - (R2-CPE2) spectrum (same generator as
the impedance-fit self test), runs the full pipeline used by the app:

    fit circuit -> extract L -> remove L -> DRT -> L-curve corner -> peaks

and checks that (a) the L-curve corner gives a sensible k_reg, (b) the DRT
reconstruction matches the L-corrected data, (c) the recovered peak
resistances add up to R1+R2 and the peak frequencies land near 1/(2 pi R Q)^(1/n).

Run:  python selftest_drt.py
"""
import numpy as np

from impedance_fit import (
    parse_z_file, fit_impedance, remove_inductance, circuit_impedance,
)
from drt import (
    make_tau_grid, compute_lcurve, solve_drt, find_peaks, default_lambda_grid,
)

TRUE = dict(L=1.0e-6, Rs=10.0,
            R1=80.0, Q1=1.0e-4, n1=0.9,
            R2=250.0, Q2=5.0e-3, n2=0.85)


def build_z_text(noise_frac: float = 0.01, seed: int = 0):
    """Synthetic ZPlot text. `noise_frac` adds Gaussian noise scaled by |Z| to
    both components — the L-curve criterion needs a real misfit floor, so a
    noiseless spectrum is not a representative test."""
    f = np.logspace(6, -1, 70)
    w = 2 * np.pi * f
    jw = 1j * w
    Z = 1j * w * TRUE["L"] + TRUE["Rs"]
    Z += TRUE["R1"] / (1 + TRUE["R1"] * TRUE["Q1"] * jw ** TRUE["n1"])
    Z += TRUE["R2"] / (1 + TRUE["R2"] * TRUE["Q2"] * jw ** TRUE["n2"])
    if noise_frac:
        rng = np.random.default_rng(seed)
        s = noise_frac * np.abs(Z)
        Z = (Z.real + rng.normal(0, s)) + 1j * (Z.imag + rng.normal(0, s))
    lines = ["ZPlot2- test", "End Comments"]
    for i, (fi, zi) in enumerate(zip(f, Z)):
        lines.append(f"{i},{fi:.8g},0,0,{zi.real:.8g},{zi.imag:.8g}")
    return "\n".join(lines)


def cpe_peak_freq(R, Q, n):
    """Characteristic frequency of an R-CPE arc: (RQ)^(-1/n) / (2 pi)."""
    return (R * Q) ** (-1.0 / n) / (2 * np.pi)


def main():
    data = parse_z_file(build_z_text())
    print(f"loaded {len(data)} points, f = {data.freq.min():.3g} .. {data.freq.max():.3g} Hz")

    # 1) circuit fit + inductance removal --------------------------------- #
    res = fit_impedance(data, n_elem=2, weighting="modulus")
    corr = remove_inductance(data, res, weighting="modulus")
    L = corr.L
    dcorr = corr.data_corr
    print(f"circuit fit: L = {L:.3e} H (true {TRUE['L']:.1e}),  "
          f"Rs = {res.params[1]:.4g},  chi2/dof = {res.chi_square_reduced:.2e}")

    # 2) DRT L-curve ------------------------------------------------------ #
    tau = make_tau_grid(dcorr.freq, ppd=10, extend_decades=1.0)
    lams = default_lambda_grid(dcorr, tau, weighting="unit")
    lc = compute_lcurve(dcorr, tau, lambdas=lams, order=1, weighting="unit")
    print(f"tau grid: {len(tau)} pts, {tau.min():.2e} .. {tau.max():.2e} s")
    print(f"L-curve: scanned {len(lams)} lambda in "
          f"[{lams.min():.2e}, {lams.max():.2e}], corner k_reg = {lc.lam_opt:.3e} "
          f"(index {lc.i_opt}/{len(lams)-1})")

    corner_interior = 0 < lc.i_opt < len(lams) - 1

    # 3) solution at the corner ------------------------------------------ #
    sol = solve_drt(dcorr, tau, lc.lam_opt, order=1, weighting="unit")

    # reconstruction quality vs the L-corrected data
    zrec = sol.z_model
    rel = np.linalg.norm(zrec - dcorr.z) / np.linalg.norm(dcorr.z) * 100
    print(f"DRT reconstruction rel. error vs L-corrected data: {rel:.3f}%")

    # 4) peaks ------------------------------------------------------------ #
    peaks = find_peaks(sol)
    print(f"\nfound {len(peaks)} peak(s):")
    f1 = cpe_peak_freq(TRUE["R1"], TRUE["Q1"], TRUE["n1"])
    f2 = cpe_peak_freq(TRUE["R2"], TRUE["Q2"], TRUE["n2"])
    print(f"  expected peaks near f1={f1:.3g} Hz (R={TRUE['R1']}), "
          f"f2={f2:.3g} Hz (R={TRUE['R2']})")
    for p in peaks:
        print(f"  f={p.freq:10.4g} Hz  R={p.resistance:8.4g} Ohm  "
              f"C={p.capacitance:.3g} F  height={p.gamma:.3g}")

    R_total_true = TRUE["R1"] + TRUE["R2"]
    R_total_drt = sum(p.resistance for p in peaks)
    r_pol_err = abs(sol.r_pol - R_total_true) / R_total_true * 100
    print(f"\ntotal R_pol: area={sol.r_pol:.4g}, peaks sum={R_total_drt:.4g}, "
          f"true={R_total_true} (area err {r_pol_err:.2f}%)")

    # ---- checks --------------------------------------------------------- #
    # NB: the low-frequency arc (f2 ~ 0.12 Hz) sits at the very edge of the
    # measured band (down to 0.1 Hz), so its resistance is inherently
    # overestimated (its tail is unobserved). We therefore check the total
    # R_pol loosely and verify R1 (the fully-observed high-f arc) tightly.
    checks = {
        "corner interior": corner_interior,
        "reconstruction < 5%": rel < 5.0,
        "two peaks found": len(peaks) == 2,
        "R_pol area within 30%": r_pol_err < 30.0,
    }
    if len(peaks) == 2:
        # peaks are high-f first; match to f1 (high) and f2 (low)
        pf_hi, pf_lo = peaks[0].freq, peaks[1].freq
        checks["peak freqs near truth"] = (0.25 < pf_hi / f1 < 4.0
                                           and 0.25 < pf_lo / f2 < 4.0)
        r1_err = abs(peaks[0].resistance - TRUE["R1"]) / TRUE["R1"] * 100
        print(f"high-f peak R1: {peaks[0].resistance:.4g} vs true {TRUE['R1']} "
              f"(err {r1_err:.1f}%)")
        checks["R1 within 15%"] = r1_err < 15.0

    print("\n-- checks --")
    ok = True
    for name, passed in checks.items():
        print(f"  [{'OK ' if passed else 'FAIL'}] {name}")
        ok = ok and passed

    print("\nRESULT:", "PASS" if ok else "CHECK")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
