import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from benford.core import analyze
from benford.data_loader import (
    load_csv,
    generate_population_data,
    generate_fake_data,
    generate_fraud_data,
)

st.set_page_config(page_title="벤포드 법칙 분석기", page_icon="🔍", layout="wide")

st.title("벤포드 법칙 분석기 — 숫자 조작 탐지 도구")
st.markdown(
    """
벤포드 법칙(Benford's Law)은 자연 발생 수치 데이터에서 첫 번째 유효숫자가 균등하게 분포하지 않고,
숫자 1이 약 30.1%로 가장 많이 나타나며 9로 갈수록 빈도가 감소한다는 통계적 법칙입니다.
이 법칙은 회계 부정, 선거 조작, 통계 위조 등을 탐지하는 데 실제로 활용됩니다.
"""
)

st.sidebar.header("데이터 입력")
mode = st.sidebar.radio(
    "데이터 입력 방식",
    [
        "직접 입력",
        "샘플 데이터: 인구 (정상)",
        "샘플 데이터: 조작된 숫자",
        "샘플 데이터: 균등 난수",
        "CSV 파일 업로드",
    ],
)

numbers = []

if mode == "직접 입력":
    raw = st.sidebar.text_area(
        "쉼표로 구분된 숫자를 입력하세요",
        value="123, 456, 789, 111, 234, 567, 890, 112, 345, 678",
        height=150,
    )
    try:
        numbers = [float(x.strip()) for x in raw.split(",") if x.strip()]
    except ValueError:
        st.sidebar.error("숫자만 입력해 주세요.")

elif mode == "샘플 데이터: 인구 (정상)":
    numbers = generate_population_data(500)

elif mode == "샘플 데이터: 조작된 숫자":
    numbers = generate_fraud_data(500)

elif mode == "샘플 데이터: 균등 난수":
    numbers = generate_fake_data(500)

elif mode == "CSV 파일 업로드":
    uploaded = st.sidebar.file_uploader("CSV 파일 선택", type=["csv"])
    if uploaded:
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        numbers = load_csv(tmp_path)

if not numbers:
    st.info("사이드바에서 데이터를 입력하거나 샘플 데이터를 선택하세요.")
    st.stop()

result = analyze(numbers)

st.markdown(f"**데이터 수 (n):** {result['n']:,}개")
st.markdown("---")

st.subheader("분석 결과")

if result["p_value"] > 0.05:
    st.success(f"✅ {result['verdict']}")
else:
    st.error(f"⚠️ {result['verdict']}")

col1, col2 = st.columns(2)
col1.metric("카이제곱 통계량 (χ²)", f"{result['chi2_stat']:.4f}")
col2.metric("p-값", f"{result['p_value']:.4f}")

st.markdown("---")
st.subheader("앞자리 숫자 분포 비교")

digits = list(range(1, 10))
obs_pct = [result["observed_pct"][d] * 100 for d in digits]
exp_pct = [result["expected_pct"][d] * 100 for d in digits]

fig = go.Figure()
fig.add_trace(
    go.Bar(
        name="실제 분포",
        x=[str(d) for d in digits],
        y=obs_pct,
        marker_color="steelblue",
    )
)
fig.add_trace(
    go.Bar(
        name="벤포드 예상",
        x=[str(d) for d in digits],
        y=exp_pct,
        marker_color="tomato",
        opacity=0.7,
    )
)
fig.update_layout(
    barmode="group",
    xaxis_title="앞자리 숫자 (1~9)",
    yaxis_title="비율 (%)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=420,
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("상세 비교표")
table_data = {
    "앞자리 숫자": digits,
    "실제 빈도": [result["observed"][d] for d in digits],
    "실제 비율 (%)": [f"{result['observed_pct'][d]*100:.2f}" for d in digits],
    "벤포드 예상 (%)": [f"{result['expected_pct'][d]*100:.2f}" for d in digits],
}
st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

with st.expander("수학 원리 보기"):
    st.markdown("### 벤포드 법칙 공식")
    st.latex(r"P(d) = \log_{10}\left(1 + \frac{1}{d}\right), \quad d \in \{1, 2, \ldots, 9\}")
    st.markdown(
        "여기서 $d$는 앞자리 숫자입니다. 예를 들어 $P(1) = \\log_{10}(2) \\approx 0.301$, "
        "즉 약 30.1%의 확률로 숫자 1이 첫 자리에 나타납니다."
    )
    st.markdown("### 카이제곱 검정 공식")
    st.latex(r"\chi^2 = \sum_{d=1}^{9} \frac{(O_d - E_d)^2}{E_d}")
    st.markdown(
        "$O_d$는 관측된 빈도, $E_d = n \\cdot P(d)$는 기댓값입니다. "
        "p-값이 0.05보다 크면 벤포드 법칙을 따른다고 판단합니다."
    )
    st.markdown("### 왜 자연 발생 데이터는 벤포드 법칙을 따를까요?")
    st.markdown(
        "자연 발생 데이터는 로그 스케일에서 균등하게 분포하는 경향이 있습니다. "
        "10배 범위(예: 1~10, 10~100)에서 각 자릿수가 차지하는 비율이 로그 비율로 결정되며, "
        "이는 스케일 불변성(scale invariance)으로 인해 단위를 바꿔도 동일한 분포가 유지됩니다."
    )

with st.expander("실제 활용 사례"):
    st.markdown(
        """
### 벤포드 법칙의 실제 활용

- **회계 부정 감사**: 기업의 재무 데이터나 경비 청구서에서 조작된 숫자를 탐지합니다.
- **선거 결과 검증**: 투표 수 데이터가 벤포드 법칙을 따르는지 분석하여 부정 선거 가능성을 평가합니다.
- **코로나19 통계 분석**: 각국의 확진자 수 보고 데이터의 신뢰성을 검토합니다.
- **세금 신고 검증**: 국세청에서 소득 신고 데이터의 이상 여부를 탐지합니다.
- **과학 데이터 검증**: 논문에서 보고된 실험 수치의 조작 여부를 확인합니다.

> **주의**: 벤포드 법칙 위반이 반드시 조작을 의미하지는 않습니다. 특정 범위로 제한된 데이터
> (예: 시급, 나이)나 데이터 수가 적은 경우에는 법칙이 적용되지 않을 수 있습니다.
> 통계적 증거는 추가 조사의 시작점으로 활용해야 합니다.
"""
    )
