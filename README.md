# DRT 분석 프로그램

임피던스(EIS) 스펙트럼으로부터 **완화시간 분포(Distribution of Relaxation Times, DRT)** 를
계산하는 프로그램입니다. 지난 임피던스 피팅 프로그램을 기반으로 하며,
**L-curve 기준**으로 정규화 파라미터(k_reg)를 결정합니다.

> 참조: Choi, Shin, Ji, H. Kim, Son, J.-H. Lee, B.-K. Kim, H.-W. Lee, Yoon,
> *"Interpretation of Impedance Spectra of Solid Oxide Fuel Cells: L-Curve
> Criterion for Determination of Regularization Parameter in Distribution
> Function of Relaxation Times Technique"*, **JOM 71(11), 3825 (2019).**

## 분석 흐름 (논문 방법 그대로)

1. **등가회로 피팅** — `Rs–(R₁-CPE₁)–(R₂-CPE₂)–…`(+직렬 인덕턴스 L)로 피팅
   (지난 프로그램과 동일한 방식, `impedance_fit.py` 재사용).
2. **인덕턴스 L 추출·제거** — jωL 을 허수부에서 빼서 **L 없는 스펙트럼** 생성.
3. **DRT** — L 없는 스펙트럼에 대해 Tikhonov 정규화(비음수 제약)로
   `Z_pol(ω) = R∞ + ∫ γ(τ)/(1+jωτ) dτ` 를 풀이.
4. **L-curve** — 여러 k_reg 에 대해 DRT 를 풀어
   `log(solution norm η)` vs `log(misfit norm ρ)` 를 그리고,
   **최대 곡률 모서리(corner)** 에서 최적 k_reg 결정 (논문 Eq.3, Eq.4, Fig.5).
5. **k_reg 조절** — 최적값을 기본으로, 배수를 로그 스케일로 바꿔가며 DRT 를
   다시 실행해 분포 변화를 관찰 (논문 Fig.4: 크면 과평활, 작으면 과적합·인공 피크).

## 실행

```bash
cd "F:/프로그램/DRT"
streamlit run app.py
```

브라우저가 열리면 좌측 사이드바에서 `.z` 파일(또는 `freq, Z', Z''` 텍스트/CSV)을
업로드하고, **① 피팅 + L 제거 실행** → **② L-curve** → **③ DRT 실행** 순서로 진행합니다.

데모용 합성 데이터: `sample_drt.z` (1% 잡음, 두 개의 R-CPE 아크).

## 파일 구성

| 파일 | 내용 |
|------|------|
| `app.py` | Streamlit UI (피팅→L제거→DRT→L-curve→k_reg 조절) |
| `drt.py` | DRT 코어 (커널·Tikhonov NNLS 풀이·L-curve·코너 검출·피크 분석) |
| `impedance_fit.py` | 등가회로 피팅·인덕턴스 제거 (지난 프로그램에서 가져옴) |
| `selftest_drt.py` | 합성 데이터로 전체 파이프라인 검증 (`python selftest_drt.py`) |
| `sample_drt.z` | 데모용 합성 스펙트럼 |

## DRT 설정 (사이드바)

- **정규화 차수 (order)**: `0차`는 논문의 solution norm = √Σγ² 과 동일,
  `1차/2차 도함수`는 더 매끄러운 분포. 기본값 1차.
- **DRT 가중치**: `modulus`(1/|Z|, EIS 표준) 또는 `unit`(균등).
- **τ 격자 밀도 / 범위 확장**: 완화시간 격자 τ = 1/(2πf) 의 해상도와,
  측정 대역 양쪽으로의 확장 폭(피크 꼬리 지지용).
- **L-curve λ 스캔 개수**: L-curve 를 그릴 k_reg 표본 수.

## 핵심 수식

- 커널(이산화, 로그-τ 격자): `Z(ω_n) = R∞ + jω_n L + Σ_m g_m·Δlnτ /(1+jω_n τ_m)`,
  여기서 `g_m = R_pol·γ(τ_m)·τ_m` (단위 Ω) 이며 **피크 면적 = 저항**.
  DRT 커널 `1/(1+jωτ)` 는 용량성 성분만 표현하므로, **R∞(오믹)·L(잔여 인덕턴스)**
  항을 자유 파라미터로 추가한다. 특히 L 은 `±ω` 두 비음수 열로 넣어 부호 자유롭게
  하여, L 제거 후 남은 유도성 꼬리를 DRT 가 흡수 — 이것이 없으면 유도성 점이
  misfit 바닥을 높여 **L-curve 코너가 과평활로 붕괴**한다.
- 비음수 정규화 최소자승: `min ‖[A; k_reg·L_reg] x − [b; 0]‖²`, `x ≥ 0` (scipy `nnls`).
- solution norm `η = ‖L_reg x‖` (Eq.3), misfit norm `ρ = ‖W(Ax−b)‖/√N` (Eq.4).
  가중치 기본값은 **unit(균등)** — 논문 Eq.4 와 동일하며, 저임피던스 시료에서
  modulus(1/|Z|) 가중치가 유발하는 과평활을 피한다.
- k_reg 선택(**misfit 무릎, knee**): EIS-DRT 는 작은 k_reg 구간에서 misfit 이 잡음
  바닥에 평평하게 눌려 순수 곡률-코너가 과평활 지점을 잘못 고른다. 대신 **misfit 이
  바닥의 (1+tol)배(기본 tol=5%) 이내인 가장 큰 k_reg** 를 택해, 피팅 품질을 해치지
  않으면서 최대한 매끄럽게 만든다(가짜 진동만 제거하고 실제 피크는 유지).
- 피크: g(ln τ) 의 골–골 구간 적분 = 저항, `C = τ_peak / R`. 총 저항의 5% 미만
  피크는 수치 잔물결로 보고 제외.

## 검증

```bash
python selftest_drt.py
```

합성 스펙트럼(1% 잡음, R₁=80Ω@34Hz, R₂=250Ω@0.12Hz)에서
L-curve 코너가 내부에 잡히고, 재구성 오차 < 5%, 두 피크 검출,
고주파 피크 저항 오차 < 15% 임을 확인합니다.
(저주파 아크는 측정 대역 끝(0.1Hz)에 걸쳐 저항이 다소 과대평가되는데,
이는 미측정 꼬리 때문으로 알고리즘 오류가 아닙니다.)
