import math
from scipy import stats


def extract_leading_digit(n):
    n = abs(float(n))
    if n == 0:
        return None
    while n < 1:
        n *= 10
    while n >= 10:
        n /= 10
    return int(n)


def benford_expected():
    return {d: math.log10(1 + 1 / d) for d in range(1, 10)}


def analyze(numbers):
    valid = [extract_leading_digit(x) for x in numbers]
    valid = [d for d in valid if d is not None]
    n = len(valid)

    observed = {d: 0 for d in range(1, 10)}
    for d in valid:
        observed[d] += 1

    expected_pct = benford_expected()
    observed_pct = {d: observed[d] / n for d in range(1, 10)}

    observed_counts = [observed[d] for d in range(1, 10)]
    expected_counts = [n * expected_pct[d] for d in range(1, 10)]

    chi2_stat, p_value = stats.chisquare(f_obs=observed_counts, f_exp=expected_counts)

    verdict = "정상 (벤포드 법칙 부합)" if p_value > 0.05 else "조작 의심 (벤포드 법칙 위반)"

    return {
        "observed": observed,
        "observed_pct": observed_pct,
        "expected_pct": expected_pct,
        "chi2_stat": chi2_stat,
        "p_value": p_value,
        "verdict": verdict,
        "n": n,
    }
