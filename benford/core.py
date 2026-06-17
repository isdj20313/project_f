"""벤포드 법칙(Benford's Law) 분석 핵심 모듈.

이 모듈은 다음 기능을 제공한다.
1. 숫자에서 첫 번째 유효숫자(맨 앞자리)를 뽑는 기능
2. 벤포드 법칙이 예측하는 이론적 분포 계산
3. 실제 데이터의 첫자리 분포 계산
4. 카이제곱 적합도 검정으로 "벤포드를 따르는가"를 통계적으로 판정

수학적 배경
-----------
벤포드 법칙: 자연스럽게 발생한 데이터의 첫자리 d가 나타날 확률은

    P(d) = log10(1 + 1/d),   d = 1, 2, ..., 9

카이제곱 적합도 검정:

    chi^2 = sum_d ( (관측빈도_d - 기대빈도_d)^2 / 기대빈도_d )

자유도 8(=9개 범주-1)에서 임계값(유의수준 5%)은 약 15.507이다.
chi^2 값이 임계값보다 크면 "유의미하게 다름 = 조작 의심 신호"로 본다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# scipy가 있으면 정확한 p-value / 임계값을 쓰고, 없으면 표 값으로 대체한다.
try:
    from scipy import stats as _scipy_stats
except ImportError:  # pragma: no cover - scipy는 requirements에 포함되어 있음
    _scipy_stats = None

# 자유도 8, 유의수준 0.05에서의 카이제곱 임계값 (scipy가 없을 때 사용)
CHI2_CRITICAL_DF8_005 = 15.50731305586545

DIGITS = (1, 2, 3, 4, 5, 6, 7, 8, 9)


def benford_probability(digit: int) -> float:
    """첫자리 ``digit``(1~9)이 나타날 벤포드 이론 확률을 돌려준다."""
    if digit < 1 or digit > 9:
        raise ValueError("첫자리는 1과 9 사이여야 합니다.")
    return math.log10(1 + 1 / digit)


def benford_expected_proportions() -> dict[int, float]:
    """1~9 각 첫자리에 대한 벤포드 이론 비율 딕셔너리."""
    return {d: benford_probability(d) for d in DIGITS}


def first_digit(value: float) -> int | None:
    """숫자 ``value``의 첫 번째 유효숫자(1~9)를 돌려준다.

    예) 4823 -> 4,  0.0071 -> 7,  -38.5 -> 3
    0이거나 숫자가 아니면 ``None``을 돌려준다(분석에서 제외).
    """
    try:
        x = abs(float(value))
    except (TypeError, ValueError):
        return None

    if x == 0 or math.isnan(x) or math.isinf(x):
        return None

    # 과학적 표기법으로 변환하면 가수부의 첫 글자가 곧 첫 유효숫자다.
    # (예: 0.3 -> '3.0...e-01', 4823 -> '4.8...e+03')
    # 이 방식은 0.3/0.1 = 2.999.. 같은 부동소수점 오차에 영향받지 않는다.
    mantissa = f"{x:.15e}"  # 충분한 자릿수로 반올림 왜곡 최소화
    digit = int(mantissa[0])
    if digit == 0:  # 정상적인 경우 발생하지 않지만 방어적으로 처리
        return None
    return digit


def extract_first_digits(values) -> list[int]:
    """반복 가능한 숫자 묶음에서 유효한 첫자리만 추출한 리스트."""
    digits = []
    for v in values:
        d = first_digit(v)
        if d is not None:
            digits.append(d)
    return digits


def observed_counts(values) -> dict[int, int]:
    """1~9 첫자리별 관측 빈도(개수) 딕셔너리. 없는 자리는 0."""
    counts = {d: 0 for d in DIGITS}
    for d in extract_first_digits(values):
        counts[d] += 1
    return counts


def observed_proportions(values) -> dict[int, float]:
    """1~9 첫자리별 관측 비율 딕셔너리. 데이터가 없으면 모두 0."""
    counts = observed_counts(values)
    total = sum(counts.values())
    if total == 0:
        return {d: 0.0 for d in DIGITS}
    return {d: counts[d] / total for d in DIGITS}


@dataclass
class BenfordResult:
    """벤포드 분석 결과를 담는 자료구조."""

    n: int  # 분석에 사용된 유효 숫자 개수
    observed_counts: dict[int, int]
    observed_proportions: dict[int, float]
    expected_proportions: dict[int, float]
    expected_counts: dict[int, float]
    chi_square: float
    degrees_of_freedom: int
    critical_value: float
    p_value: float | None
    mad: float  # 평균 절대 편차(Mean Absolute Deviation)
    follows_benford: bool  # True면 벤포드를 따른다(정상)로 판정
    verdict: str = field(default="")

    def summary(self) -> str:
        """사람이 읽기 좋은 한 줄 요약 문자열."""
        status = "정상(벤포드를 따름)" if self.follows_benford else "조작 의심(벤포드에서 벗어남)"
        p_text = f"{self.p_value:.4f}" if self.p_value is not None else "N/A"
        return (
            f"표본 {self.n}개 | chi^2={self.chi_square:.3f} "
            f"(임계값 {self.critical_value:.3f}, p={p_text}) -> {status}"
        )


def analyze(values, alpha: float = 0.05) -> BenfordResult:
    """숫자 묶음을 받아 벤포드 카이제곱 분석을 수행한다.

    Parameters
    ----------
    values : iterable
        분석할 숫자들(문자열/실수/정수 혼합 가능, 0과 비숫자는 제외).
    alpha : float
        유의수준(기본 0.05). 카이제곱 임계값과 판정 기준에 사용.

    Returns
    -------
    BenfordResult
    """
    counts = observed_counts(values)
    n = sum(counts.values())

    expected_prop = benford_expected_proportions()
    obs_prop = {d: (counts[d] / n if n else 0.0) for d in DIGITS}
    expected_counts = {d: expected_prop[d] * n for d in DIGITS}

    # 카이제곱 통계량
    chi_square = 0.0
    for d in DIGITS:
        e = expected_counts[d]
        if e > 0:
            chi_square += (counts[d] - e) ** 2 / e

    dof = len(DIGITS) - 1  # 8

    # 임계값과 p-value
    if _scipy_stats is not None:
        critical_value = float(_scipy_stats.chi2.ppf(1 - alpha, dof))
        p_value = float(_scipy_stats.chi2.sf(chi_square, dof))
    else:
        critical_value = CHI2_CRITICAL_DF8_005
        p_value = None

    # 평균 절대 편차(MAD): 첫자리 분석에서 널리 쓰는 보조 지표
    mad = sum(abs(obs_prop[d] - expected_prop[d]) for d in DIGITS) / len(DIGITS)

    # 판정: 표본이 너무 적으면 검정 신뢰도가 낮으므로 따로 표시
    follows = chi_square <= critical_value

    if n < 30:
        verdict = "표본이 너무 적어(30개 미만) 검정 신뢰도가 낮습니다. 참고용으로만 보세요."
    elif follows:
        verdict = "벤포드 법칙을 따릅니다. 통계적으로 자연스러운 분포입니다."
    else:
        verdict = (
            "벤포드 법칙에서 유의미하게 벗어났습니다. 추가 검증이 필요한 신호입니다. "
            "(이 결과만으로 조작이라고 단정할 수는 없습니다.)"
        )

    return BenfordResult(
        n=n,
        observed_counts=counts,
        observed_proportions=obs_prop,
        expected_proportions=expected_prop,
        expected_counts=expected_counts,
        chi_square=chi_square,
        degrees_of_freedom=dof,
        critical_value=critical_value,
        p_value=p_value,
        mad=mad,
        follows_benford=follows,
        verdict=verdict,
    )
