"""벤포드 법칙 인터랙티브 대시보드 (Streamlit).

실행:
    streamlit run app.py

사람들이 직접 데이터를 넣어 보고, 실제 분포와 벤포드 예상 분포를 비교하며,
카이제곱 검정으로 '정상 / 조작 의심'을 체험할 수 있는 웹 대시보드다.
"""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from benford import analyze
from benford.core import DIGITS
from benford.datasets import get_named_datasets

st.set_page_config(page_title="벤포드 법칙 탐지기", page_icon="🔢", layout="wide")


def parse_pasted_numbers(text: str) -> list[float]:
    """줄/쉼표/공백으로 구분된 텍스트에서 숫자를 뽑아낸다."""
    values: list[float] = []
    for token in text.replace(",", " ").split():
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values


def comparison_dataframe(result) -> pd.DataFrame:
    """관측 비율 vs 벤포드 예상 비율 비교용 데이터프레임."""
    rows = []
    for d in DIGITS:
        rows.append(
            {
                "첫자리": d,
                "관측 비율(%)": round(result.observed_proportions[d] * 100, 2),
                "벤포드 예상(%)": round(result.expected_proportions[d] * 100, 2),
                "관측 개수": result.observed_counts[d],
            }
        )
    return pd.DataFrame(rows).set_index("첫자리")


# ---------------------------------------------------------------------------
# 사이드바: 데이터 입력 방법 선택
# ---------------------------------------------------------------------------
st.sidebar.title("🔢 데이터 입력")
st.sidebar.caption("아래에서 분석할 데이터를 고르거나 직접 넣어 보세요.")

input_mode = st.sidebar.radio(
    "입력 방법",
    ["예시 데이터", "직접 입력(붙여넣기)", "CSV 업로드"],
)

values: list[float] = []
title = ""

datasets = get_named_datasets()

if input_mode == "예시 데이터":
    choice = st.sidebar.selectbox("예시 데이터 선택", list(datasets.keys()))
    values = datasets[choice]
    title = choice

elif input_mode == "직접 입력(붙여넣기)":
    st.sidebar.caption("숫자를 줄바꿈/쉼표/공백으로 구분해 넣으세요.")
    pasted = st.sidebar.text_area(
        "숫자 입력",
        height=200,
        placeholder="예)\n4823\n1290\n57\n0.0071\n...",
    )
    values = parse_pasted_numbers(pasted)
    title = "직접 입력한 데이터"

else:  # CSV 업로드
    uploaded = st.sidebar.file_uploader("CSV 파일 업로드", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception:  # noqa: BLE001 - 인코딩 문제 등 대비
            uploaded.seek(0)
            df = pd.read_csv(io.StringIO(uploaded.getvalue().decode("utf-8-sig")))
        numeric_cols = [c for c in df.columns
                        if pd.api.types.is_numeric_dtype(df[c])] or list(df.columns)
        column = st.sidebar.selectbox("분석할 열 선택", numeric_cols)
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        values = series.tolist()
        title = f"{uploaded.name} - {column}"

alpha = st.sidebar.slider("유의수준 (α)", 0.01, 0.10, 0.05, 0.01,
                          help="카이제곱 검정의 기준 확률. 보통 0.05를 씁니다.")

# ---------------------------------------------------------------------------
# 본문
# ---------------------------------------------------------------------------
st.title("벤포드 법칙으로 숫자 데이터 검증하기")
st.markdown(
    "자연스럽게 발생한 데이터의 **맨 앞자리**는 1이 약 30%로 가장 많고 9로 갈수록 "
    "적어집니다. 이 분포에서 크게 벗어나면 **조작이 의심되는 신호**일 수 있습니다. "
    "왼쪽에서 데이터를 골라 직접 확인해 보세요."
)

with st.expander("📐 벤포드 법칙과 카이제곱 검정이 뭔가요? (수학 원리)"):
    st.markdown(
        r"""
**벤포드 법칙**: 첫자리 $d$가 나타날 확률은

$$P(d) = \log_{10}\!\left(1 + \frac{1}{d}\right), \quad d = 1, \dots, 9$$

여러 자릿수에 걸친 데이터는 로그 스케일에서 고르게 퍼지기 때문에,
1로 시작하는 구간에 가장 오래 머물러 1이 가장 많아집니다.

**카이제곱 적합도 검정**: 실제 분포가 벤포드에서 벗어난 정도를 통계량으로 잰다.

$$\chi^2 = \sum_{d=1}^{9} \frac{(\text{관측빈도}_d - \text{기대빈도}_d)^2}{\text{기대빈도}_d}$$

자유도 8에서 이 값이 임계값(α=0.05일 때 약 15.51)보다 크면
"유의미하게 다르다 = 추가 검증이 필요한 신호"로 봅니다.
        """
    )

if not values:
    st.info("👈 왼쪽 사이드바에서 데이터를 선택하거나 입력하면 결과가 나타납니다.")
    st.stop()

result = analyze(values, alpha=alpha)

# --- 핵심 지표 ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("분석한 숫자 개수", f"{result.n:,}")
col2.metric("카이제곱 통계량", f"{result.chi_square:.2f}")
col3.metric(f"임계값 (α={alpha})", f"{result.critical_value:.2f}")
if result.p_value is not None:
    col4.metric("p-value", f"{result.p_value:.4f}")
else:
    col4.metric("MAD", f"{result.mad:.4f}")

# --- 판정 배너 ---
if result.n < 30:
    st.warning(f"⚠️ {result.verdict}")
elif result.follows_benford:
    st.success(f"✅ 정상: {result.verdict}")
else:
    st.error(f"🚨 조작 의심: {result.verdict}")

# --- 그래프 + 표 ---
left, right = st.columns([3, 2])

comp_df = comparison_dataframe(result)

with left:
    st.subheader("관측 분포 vs 벤포드 예상 분포")
    chart_df = comp_df[["관측 비율(%)", "벤포드 예상(%)"]]
    st.bar_chart(chart_df, height=380)
    st.caption("두 막대가 자리마다 비슷할수록 벤포드 법칙을 잘 따르는 것입니다.")

with right:
    st.subheader("자리별 상세 표")
    st.dataframe(comp_df, width="stretch", height=380)

# --- 결과 요약 텍스트 (발표/보고서에 복사용) ---
with st.expander("📋 결과 요약 (복사해서 보고서에 붙여넣기)"):
    st.code(result.summary(), language="text")

st.divider()
st.caption(
    "ℹ️ 벤포드 법칙은 '신호'일 뿐 증거가 아닙니다. 벗어났다고 해서 조작이라고 "
    "단정할 수 없으며, 추가 검증이 필요한 대상을 찾는 1차 선별 도구로 쓰입니다."
)
