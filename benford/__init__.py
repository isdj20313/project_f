"""벤포드 법칙 분석 패키지.

핵심 분석 기능은 :mod:`benford.core`, 예시 데이터는 :mod:`benford.datasets`에 있다.
"""

from .core import (
    BenfordResult,
    analyze,
    benford_expected_proportions,
    benford_probability,
    extract_first_digits,
    first_digit,
    observed_counts,
    observed_proportions,
)

__all__ = [
    "BenfordResult",
    "analyze",
    "benford_expected_proportions",
    "benford_probability",
    "extract_first_digits",
    "first_digit",
    "observed_counts",
    "observed_proportions",
]

__version__ = "1.0.0"
