"""
Streamlit UI for Distribution of Relaxation Times (DRT) analysis.

Pipeline (based on Choi et al., JOM 71, 3825 (2019)):

    1. Fit the equivalent circuit  Rs - (R1-CPE1) - (R2-CPE2) - ...  (+ series L)
    2. Extract the inductance L and remove it  ->  L-corrected spectrum
    3. DRT (Tikhonov regularization, non-negative) on the L-corrected spectrum
    4. L-curve criterion:  log(solution norm) vs log(misfit norm),
       optimum k_reg at the maximum-curvature corner
    5. Run the DRT at that k_reg; also sweep k_reg around it (paper Fig. 4)

Run with:   streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from impedance_fit import (
    parse_z_file, preview_columns, suggest_columns,
    fit_impedance, circuit_impedance, remove_inductance,
    detect_hf_artifact, subset,
)
from drt import (
    make_tau_grid, default_lambda_grid, compute_lcurve, solve_drt, find_peaks,
)

st.set_page_config(page_title="DRT 분석", layout="wide")

st.title("DRT 분석 (Distribution of Relaxation Times)")
st.markdown(
    "임피던스 스펙트럼을 **등가회로 Rs–(R₁-CPE₁)–(R₂-CPE₂)–…** 로 피팅해 직렬 "
    "인덕턴스 $L$ 을 추출·제거한 뒤, 인덕턴스 없는 스펙트럼으로 **DRT** 를 수행합니다.\n\n"
    r"$Z_{pol}(\omega) = R_\infty + \displaystyle\int_0^\infty "
    r"\frac{\gamma(\tau)}{1+j\omega\tau}\,d\tau$  "
    "— Tikhonov 정규화(비음수), **L-curve 기준**으로 정규화 파라미터 $k_{reg}$ 결정.\n\n"
    "*참조: Choi et al., JOM 71(11), 3825 (2019).*"
)

PLOT_CONFIG = {"scrollZoom": True, "displaylogo": False,
               "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


# --------------------------------------------------------------------------- #
#  Plot helpers (shared style with the sibling impedance-fit app)
# --------------------------------------------------------------------------- #
def nyquist_fig(traces, height=460):
    """traces: list of (name, z_real, z_imag, freq, color, is_line[, symbol])."""
    fig = go.Figure()
    for tr in traces:
        name, zr, zi, freq, color, is_line = tr[:6]
        symbol = tr[6] if len(tr) > 6 else "circle"
        fig.add_trace(go.Scatter(
            x=zr, y=-np.asarray(zi), name=name,
            mode="lines" if is_line else "markers",
            line=dict(color=color, width=2),
            marker=dict(color=color, size=7, symbol=symbol),
            customdata=freq,
            hovertemplate="Z'=%{x:.4g} Ω<br>-Z''=%{y:.4g} Ω"
                          "<br>f=%{customdata:.4g} Hz<extra>" + name + "</extra>",
        ))
    fig.update_layout(height=height, hovermode="closest", dragmode="zoom",
                      margin=dict(l=60, r=20, t=10, b=50),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0))
    fig.update_xaxes(title_text="Z'  (Ω)", showgrid=True, zeroline=False)
    fig.update_yaxes(title_text="-Z''  (Ω)", showgrid=True, zeroline=False,
                     scaleanchor="x", scaleratio=1)
    return fig


def bode_fig(series, height=460):
    """series: list of (name, freq, z_complex, color, is_line[, symbol])."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07)
    for s in series:
        name, freq, z, color, is_line = s[:5]
        symbol = s[5] if len(s) > 5 else "circle"
        mode = "lines" if is_line else "markers"
        fig.add_trace(go.Scatter(
            x=freq, y=np.abs(z), name=name, mode=mode,
            line=dict(color=color, width=2), marker=dict(color=color, size=6, symbol=symbol),
            hovertemplate="f=%{x:.4g} Hz<br>|Z|=%{y:.4g} Ω<extra>" + name + "</extra>",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=freq, y=np.degrees(np.angle(z)), name=name, mode=mode,
            line=dict(color=color, width=2), marker=dict(color=color, size=6, symbol=symbol),
            showlegend=False,
            hovertemplate="f=%{x:.4g} Hz<br>phase=%{y:.3g}°<extra>" + name + "</extra>",
        ), row=2, col=1)
    fig.update_layout(height=height, hovermode="closest",
                      margin=dict(l=60, r=20, t=10, b=50),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0))
    fig.update_xaxes(type="log", showgrid=True)
    fig.update_xaxes(title_text="frequency (Hz)", row=2, col=1)
    fig.update_yaxes(type="log", title_text="|Z| (Ω)", showgrid=True, row=1, col=1)
    fig.update_yaxes(title_text="phase (°)", showgrid=True, row=2, col=1)
    return fig


def drt_fig(sols, height=460):
    """sols: list of (name, freq_tau, gamma, color[, dash]).  γ(τ) vs frequency."""
    fig = go.Figure()
    for s in sols:
        name, ft, g, color = s[:4]
        dash = s[4] if len(s) > 4 else "solid"
        fig.add_trace(go.Scatter(
            x=ft, y=g, name=name, mode="lines",
            line=dict(color=color, width=2, dash=dash),
            hovertemplate="f=%{x:.4g} Hz<br>γ=%{y:.4g} Ω<extra>" + name + "</extra>",
        ))
    fig.update_layout(height=height, hovermode="x unified",
                      margin=dict(l=60, r=20, t=10, b=50),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0))
    fig.update_xaxes(type="log", title_text="frequency  f = 1/(2πτ)  (Hz)",
                     showgrid=True, autorange="reversed")
    fig.update_yaxes(title_text="γ(τ)   (Ω per unit ln τ)", showgrid=True)
    return fig


def lcurve_fig(lc, height=460):
    """L-curve: log(misfit norm) vs log(solution norm), corner marked."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=lc.rho, y=lc.eta, mode="lines+markers", name="L-curve",
        line=dict(color="#1f77b4", width=2), marker=dict(size=5),
        customdata=lc.lambdas,
        hovertemplate="misfit ρ=%{x:.3g}<br>solution η=%{y:.3g}"
                      "<br>k_reg=%{customdata:.3g}<extra></extra>",
    ))
    io = lc.i_opt
    fig.add_trace(go.Scatter(
        x=[lc.rho[io]], y=[lc.eta[io]], mode="markers+text",
        name="corner (최적)", marker=dict(color="#d62728", size=13, symbol="star"),
        text=[f"  k_reg={lc.lam_opt:.2e}"], textposition="top right",
        hovertemplate="corner<br>k_reg=%{text}<extra></extra>",
    ))
    fig.update_layout(height=height, hovermode="closest",
                      margin=dict(l=60, r=20, t=10, b=50),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0))
    fig.update_xaxes(type="log", title_text="misfit norm  ρ  (Eq. 4)", showgrid=True)
    fig.update_yaxes(type="log", title_text="solution norm  η  (Eq. 3)", showgrid=True)
    return fig


# --------------------------------------------------------------------------- #
#  Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("1. 데이터 입력")
    up = st.file_uploader("z 파일 업로드 (.z, .txt, .csv)", type=["z", "txt", "csv", "dat"])

    st.header("2. 회로 / 인덕턴스")
    n_elem = st.number_input("(R-CPE) element 개수", 1, 8, 2, 1,
                             help="L 추출용 등가회로 Rs-(R-CPE)×N 의 원소 개수")
    fit_weighting = st.selectbox("피팅 가중치", ["modulus", "proportional", "unit"], 0)
    auto_hf = st.checkbox("고주파 꼬임 구간 자동 제외", True,
                          help="인덕턴스 꼬리에서 수직선을 벗어나 도는 고주파 이상점을 제외")
    hf_tol = st.slider("수직 판정 허용폭 (%)", 1, 30, 8, disabled=not auto_hf) / 100.0

    st.header("3. DRT 설정")
    reg_order = st.selectbox(
        "정규화 차수 (order)", [0, 1, 2], index=1,
        format_func=lambda o: {0: "0차 (‖γ‖, 논문 Eq.3)",
                               1: "1차 도함수 (기본)",
                               2: "2차 도함수 (가장 매끄러움)"}[o],
        help="0차는 논문의 solution norm=√Σγ² 과 동일. 1·2차는 더 매끄러운 분포.",
    )
    drt_weighting = st.selectbox("DRT 가중치", ["modulus", "unit"], 0,
                                 help="modulus=1/|Z| (EIS 표준), unit=균등")
    ppd = st.slider("τ 격자 밀도 (points/decade)", 5, 20, 10)
    ext = st.slider("τ 범위 확장 (decades, 양쪽)", 0.0, 2.0, 1.0, 0.5)
    n_lambda = st.slider("L-curve λ 스캔 개수", 20, 120, 60, 10)

if up is None:
    st.info("좌측에서 ZView `.z` 파일(또는 freq, Z', Z'' 텍스트 파일)을 업로드하세요.")
    st.stop()

raw = up.getvalue().decode("utf-8", errors="replace")

# --- column mapping -------------------------------------------------------- #
rows, ncol = preview_columns(raw)
with st.expander("열(column) 매핑 확인 / 수정", expanded=False):
    if rows:
        st.dataframe(pd.DataFrame(rows, columns=[str(i) for i in range(len(rows[0]))]),
                     use_container_width=True)
    d_f, d_r, d_i = suggest_columns(raw)
    st.caption(f"자동 감지된 열 → 주파수: {d_f}, Z': {d_r}, Z'': {d_i}")
    c1, c2, c3 = st.columns(3)
    col_freq = c1.number_input("주파수 열", 0, max(ncol - 1, 0), min(d_f, ncol - 1))
    col_zr = c2.number_input("Z' (실수) 열", 0, max(ncol - 1, 0), min(d_r, ncol - 1))
    col_zi = c3.number_input("Z'' (허수) 열", 0, max(ncol - 1, 0), min(d_i, ncol - 1))
    flip = st.checkbox("Z'' 부호 반전 (파일이 -Z''로 저장된 경우)", value=False)

try:
    data = parse_z_file(raw, int(col_freq), int(col_zr), int(col_zi))
except Exception as e:  # noqa: BLE001
    st.error(f"파일 파싱 오류: {e}")
    st.stop()
if flip:
    data.z_imag = -data.z_imag

st.success(f"데이터 {len(data)} 점 로드  ·  주파수 {data.freq.min():.3g} – {data.freq.max():.3g} Hz")

# --- high-frequency exclusion ---------------------------------------------- #
keep = detect_hf_artifact(data, tol_frac=hf_tol) if auto_hf else np.ones(len(data), bool)
excl = ~keep
data_fit = subset(data, keep)
if excl.any():
    st.warning(f"고주파 꼬임 **{int(excl.sum())}점** 제외 (f ≥ {data.freq[excl].min():.3g} Hz). "
               f"남은 {len(data_fit)}점으로 진행합니다.")


# --------------------------------------------------------------------------- #
#  Step A — circuit fit + inductance removal
# --------------------------------------------------------------------------- #
st.divider()
st.header("① 등가회로 피팅 → 인덕턴스(L) 제거")

run = st.button("피팅 + L 제거 실행 ▶", type="primary")
if run:
    with st.spinner("등가회로 피팅 중..."):
        fit = fit_impedance(data_fit, int(n_elem), weighting=fit_weighting)
        corr = remove_inductance(data_fit, fit, weighting=fit_weighting)
    st.session_state["fit"] = fit
    st.session_state["corr"] = corr
    st.session_state["fit_sig"] = (len(data_fit), int(n_elem), fit_weighting,
                                   float(data_fit.freq[0]), float(data_fit.z_real[0]))
    # any cached DRT/L-curve belongs to an older fit
    st.session_state.pop("lcurve", None)

fit = st.session_state.get("fit")
corr = st.session_state.get("corr")
# invalidate a stale fit (data / mapping / element count changed)
sig = (len(data_fit), int(n_elem), fit_weighting,
       float(data_fit.freq[0]) if len(data_fit) else 0.0,
       float(data_fit.z_real[0]) if len(data_fit) else 0.0)
if fit is not None and st.session_state.get("fit_sig") != sig:
    fit = corr = None
    st.info("데이터/설정이 바뀌었습니다. **‘피팅 + L 제거 실행’** 을 다시 눌러주세요.")

if fit is None:
    st.caption("‘피팅 + L 제거 실행’ 을 누르면 결과가 표시됩니다.")
    st.stop()

dcorr = corr.data_corr

m1, m2, m3, m4 = st.columns(4)
m1.metric("제거한 L (H)", f"{corr.L:.4g}")
m2.metric("Rs · 오믹 (Ω)", f"{corr.Rs:.5g}")
m3.metric("Rp 합계 (Ω)", f"{corr.Rp_total:.5g}")
m4.metric("χ²/dof", f"{fit.chi_square_reduced:.3g}")

cA, cB = st.columns(2)
with cA:
    st.markdown("**Nyquist — 인덕턴스 제거 전/후**")
    f_dense = np.logspace(np.log10(dcorr.freq.min()), np.log10(dcorr.freq.max()), 400)
    mp = corr.fit.params.copy(); mp[0] = 0.0
    z_model = circuit_impedance(mp, f_dense, corr.fit.n_elem)
    tr = [("원본 data", data_fit.z_real, data_fit.z_imag, data_fit.freq, "#1f77b4", False),
          ("L 제거 data", dcorr.z_real, dcorr.z_imag, dcorr.freq, "#2ca02c", False, "diamond"),
          ("L 제거 모델", z_model.real, z_model.imag, f_dense, "#d62728", True)]
    st.plotly_chart(nyquist_fig(tr), use_container_width=True, config=PLOT_CONFIG)
with cB:
    st.markdown("**Bode — 인덕턴스 제거 전/후**")
    ser = [("원본 data", data_fit.freq, data_fit.z, "#1f77b4", False),
           ("L 제거 data", dcorr.freq, dcorr.z, "#2ca02c", False, "diamond"),
           ("L 제거 모델", f_dense, z_model, "#d62728", True)]
    st.plotly_chart(bode_fig(ser), use_container_width=True, config=PLOT_CONFIG)


# --------------------------------------------------------------------------- #
#  Step B — L-curve
# --------------------------------------------------------------------------- #
st.divider()
st.header("② L-curve 로 정규화 파라미터 $k_{reg}$ 결정")
st.caption(
    "여러 $k_{reg}$ 에 대해 DRT 를 풀어 **solution norm η (거칠기, Eq.3)** 와 "
    "**misfit norm ρ (데이터 편차, Eq.4)** 를 로그-로그로 그립니다. "
    "**꺾인 모서리(최대 곡률)** 가 과평활–과적합의 균형점 = 최적 $k_{reg}$ 입니다."
)

tau = make_tau_grid(dcorr.freq, ppd=int(ppd), extend_decades=float(ext))
lc_sig = (sig, int(reg_order), drt_weighting, int(ppd), float(ext), int(n_lambda))
if st.session_state.get("lcurve") is None or st.session_state.get("lc_sig") != lc_sig:
    with st.spinner(f"L-curve 계산 중... ({n_lambda}개 λ 스캔)"):
        lams = default_lambda_grid(dcorr, tau, weighting=drt_weighting, n=int(n_lambda))
        lc = compute_lcurve(dcorr, tau, lambdas=lams, order=int(reg_order),
                            weighting=drt_weighting)
    st.session_state["lcurve"] = lc
    st.session_state["lc_sig"] = lc_sig
lc = st.session_state["lcurve"]

cL, cR = st.columns([1, 1])
with cL:
    st.plotly_chart(lcurve_fig(lc), use_container_width=True, config=PLOT_CONFIG)
    st.metric("최적 k_reg (L-curve corner)", f"{lc.lam_opt:.4g}")
with cR:
    # curvature vs lambda, corner marked
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lc.lambdas, y=lc.curvature, mode="lines+markers",
                             line=dict(color="#9467bd"), name="곡률"))
    fig.add_trace(go.Scatter(x=[lc.lam_opt], y=[lc.curvature[lc.i_opt]],
                             mode="markers", marker=dict(color="#d62728", size=13, symbol="star"),
                             name="corner"))
    fig.update_layout(height=460, margin=dict(l=60, r=20, t=10, b=50),
                      legend=dict(orientation="h", y=1.0))
    fig.update_xaxes(type="log", title_text="k_reg", showgrid=True)
    fig.update_yaxes(title_text="L-curve 곡률", showgrid=True)
    st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)


# --------------------------------------------------------------------------- #
#  Step C — DRT at chosen k_reg (adjustable around the optimum)
# --------------------------------------------------------------------------- #
st.divider()
st.header("③ DRT 실행 — $k_{reg}$ 조절")
st.caption(
    "기본값은 L-curve 최적 $k_{reg}$ 입니다. 아래 배수로 기본값 근처에서 조금씩 바꿔가며 "
    "분포 변화를 확인하세요 (논문 Fig.4: $k_{reg}$ 이 크면 과평활, 작으면 과적합·인공 피크)."
)

c1, c2 = st.columns([2, 1])
with c1:
    log_mult = st.slider("k_reg 배수 (log₁₀)", -3.0, 3.0, 0.0, 0.1,
                         help="최적값 × 10^(이 값). 0 = L-curve 최적값")
with c2:
    show_extra = st.checkbox("최적값도 함께 표시", True)
lam_user = lc.lam_opt * (10.0 ** log_mult)
st.markdown(f"현재 $k_{{reg}}$ = **{lam_user:.4g}**  (최적값 {lc.lam_opt:.4g} × 10^{log_mult:+.1f})")

sol = solve_drt(dcorr, tau, lam_user, order=int(reg_order), weighting=drt_weighting)
sol_opt = solve_drt(dcorr, tau, lc.lam_opt, order=int(reg_order), weighting=drt_weighting)

cD, cE = st.columns(2)
with cD:
    st.markdown("**DRT γ(τ)**")
    traces = [(f"k_reg={lam_user:.2e}", sol.freq_tau, sol.gamma, "#d62728")]
    if show_extra and abs(log_mult) > 1e-9:
        traces.append((f"최적 {lc.lam_opt:.2e}", sol_opt.freq_tau, sol_opt.gamma, "#1f77b4", "dot"))
    st.plotly_chart(drt_fig(traces), use_container_width=True, config=PLOT_CONFIG)
with cE:
    st.markdown("**재구성(inverse) vs L 제거 data** (논문 Fig.5b)")
    tr = [("L 제거 data", dcorr.z_real, dcorr.z_imag, dcorr.freq, "#2ca02c", False, "diamond"),
          (f"DRT 재구성", sol.z_model.real, sol.z_model.imag, dcorr.freq, "#d62728", True)]
    st.plotly_chart(nyquist_fig(tr), use_container_width=True, config=PLOT_CONFIG)

rel = np.linalg.norm(sol.z_model - dcorr.z) / np.linalg.norm(dcorr.z) * 100
mm1, mm2, mm3 = st.columns(3)
mm1.metric("R_∞ (오믹, Ω)", f"{sol.r_inf:.5g}")
mm2.metric("R_pol (분포 면적, Ω)", f"{sol.r_pol:.5g}")
mm3.metric("재구성 상대오차", f"{rel:.3f} %")

# --- peaks ----------------------------------------------------------------- #
peaks = find_peaks(sol)
st.markdown("**피크 분석** — 각 피크의 면적 = 저항, $C = \\tau_{peak}/R$")
if peaks:
    dfp = pd.DataFrame([{
        "peak f (Hz)": p.freq, "τ (s)": p.tau, "R (Ω)": p.resistance,
        "C (F)": p.capacitance, "height γ (Ω)": p.gamma,
    } for p in peaks])
    st.dataframe(dfp.style.format({
        "peak f (Hz)": "{:.4g}", "τ (s)": "{:.4g}", "R (Ω)": "{:.5g}",
        "C (F)": "{:.4g}", "height γ (Ω)": "{:.4g}"}), use_container_width=True)
else:
    st.info("현재 k_reg 에서 뚜렷한 피크가 검출되지 않았습니다 (과평활일 수 있음 — 배수를 낮춰보세요).")

# --- downloads ------------------------------------------------------------- #
st.divider()
st.subheader("다운로드")
drt_table = pd.DataFrame({
    "tau_s": sol.tau, "freq_Hz": sol.freq_tau, "gamma_Ohm": sol.gamma,
})
lc_table = pd.DataFrame({
    "k_reg": lc.lambdas, "misfit_norm_rho": lc.rho,
    "solution_norm_eta": lc.eta, "curvature": lc.curvature,
})
recon = pd.DataFrame({
    "freq_Hz": dcorr.freq, "Zreal_corr": dcorr.z_real, "Zimag_corr": dcorr.z_imag,
    "Zreal_recon": sol.z_model.real, "Zimag_recon": sol.z_model.imag,
})
d1, d2, d3 = st.columns(3)
d1.download_button("DRT γ(τ) CSV", drt_table.to_csv(index=False).encode("utf-8-sig"),
                   file_name=f"drt_kreg_{lam_user:.2e}.csv", mime="text/csv")
d2.download_button("L-curve CSV", lc_table.to_csv(index=False).encode("utf-8-sig"),
                   file_name="lcurve.csv", mime="text/csv")
d3.download_button("재구성 스펙트럼 CSV", recon.to_csv(index=False).encode("utf-8-sig"),
                   file_name="drt_reconstruction.csv", mime="text/csv")
if peaks:
    st.download_button("피크 표 CSV", dfp.to_csv(index=False).encode("utf-8-sig"),
                       file_name="drt_peaks.csv", mime="text/csv")
