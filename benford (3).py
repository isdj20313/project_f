"""
Benford's Law Data Verification Tool
벤포드 법칙 기반 가짜·조작 데이터 검증 도구

Python 3.9+ | stdlib + scipy (optional)
"""

import math
import re
import sys

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def extract_first_digit(n: float) -> int:
    """Return the first significant digit (1-9) of a positive number."""
    if n <= 0:
        raise ValueError(f"extract_first_digit expects a positive number, got {n}")
    # Normalise to [1, 10)
    magnitude = math.floor(math.log10(abs(n)))
    leading = abs(n) / (10 ** magnitude)
    return int(leading)


def preprocess(raw_numbers) -> list:
    """
    Accept a list of mixed str/numbers.
    - Strip currency symbols (₩, $, €, £, ¥) and % signs
    - Remove commas and whitespace
    - Discard zeros, negatives, and anything non-numeric
    Returns a list of positive floats.
    """
    result = []
    for item in raw_numbers:
        s = str(item).strip()
        # Remove currency and percent characters
        s = re.sub(r'[₩$€£¥%,\s]', '', s)
        if not s:
            continue
        try:
            val = float(s)
        except ValueError:
            continue
        if val > 0:
            result.append(val)
    return result


def parse_input(text: str) -> list:
    """
    Parse a string whose numbers are separated by commas, newlines, or spaces.
    Returns a preprocessed list of positive floats.
    """
    # Split on comma, newline, or whitespace sequences
    tokens = re.split(r'[,\n\s]+', text.strip())
    return preprocess(tokens)


# ---------------------------------------------------------------------------
# Benford statistics
# ---------------------------------------------------------------------------

def benford_expected() -> dict:
    """Return {1: P(1), …, 9: P(9)} where P(d) = log10(1 + 1/d)."""
    return {d: math.log10(1 + 1 / d) for d in range(1, 10)}


def observed_distribution(numbers: list) -> dict:
    """Return {1: count, …, 9: count} for the leading digits of numbers."""
    counts = {d: 0 for d in range(1, 10)}
    for n in numbers:
        try:
            d = extract_first_digit(n)
            if 1 <= d <= 9:
                counts[d] += 1
        except (ValueError, ZeroDivisionError):
            pass
    return counts


def chi_square_test(observed_counts: dict, total: int, expected_probs: dict):
    """
    Chi-square goodness-of-fit test against Benford's expected probabilities.

    Returns (chi2_stat: float, p_value: float, df: int).
    df is always 8 (9 categories - 1).

    Uses scipy.stats.chisquare when available; falls back to a manual
    implementation with a series-expansion p-value.
    """
    digits = list(range(1, 10))
    observed = [observed_counts.get(d, 0) for d in digits]
    expected = [expected_probs[d] * total for d in digits]

    try:
        from scipy.stats import chisquare
        chi2_stat, p_value = chisquare(f_obs=observed, f_exp=expected)
        return float(chi2_stat), float(p_value), 8
    except ImportError:
        pass

    # Manual chi-square
    chi2_stat = sum(
        (o - e) ** 2 / e for o, e in zip(observed, expected) if e > 0
    )
    p_value = _chi2_pvalue(chi2_stat, df=8)
    return float(chi2_stat), float(p_value), 8


def _regularized_gamma_p(a: float, x: float, max_iter: int = 200, tol: float = 1e-12) -> float:
    """
    Lower regularized incomplete gamma function P(a, x) via series expansion:
        P(a, x) = e^{-x} * x^a * sum_{n=0}^{inf} x^n / (a * (a+1) * … * (a+n))
    """
    if x < 0:
        raise ValueError("x must be >= 0")
    if x == 0:
        return 0.0

    # Use series for x < a + 1, continued fraction otherwise
    if x < a + 1:
        # Series
        term = 1.0 / a
        total = term
        for n in range(1, max_iter):
            term *= x / (a + n)
            total += term
            if abs(term) < tol * abs(total):
                break
        return math.exp(-x + a * math.log(x) - _log_gamma(a)) * total
    else:
        # Continued fraction for Q(a, x), then P = 1 - Q
        return 1.0 - _regularized_gamma_q_cf(a, x, max_iter, tol)


def _regularized_gamma_q_cf(a: float, x: float, max_iter: int = 200, tol: float = 1e-12) -> float:
    """
    Upper regularized incomplete gamma Q(a, x) via Lentz continued fraction.
    """
    # Lentz method
    fpmin = 1e-300
    b = x + 1.0 - a
    c = 1.0 / fpmin
    d = 1.0 / b
    h = d
    for i in range(1, max_iter + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < fpmin:
            d = fpmin
        c = b + an / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < tol:
            break
    return math.exp(-x + a * math.log(x) - _log_gamma(a)) * h


def _log_gamma(z: float) -> float:
    """log(Gamma(z)) via Lanczos approximation (accurate for z > 0)."""
    # Lanczos coefficients g=7
    g = 7
    p = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ]
    if z < 0.5:
        return math.log(math.pi / math.sin(math.pi * z)) - _log_gamma(1 - z)
    z -= 1
    x = p[0]
    for i in range(1, g + 2):
        x += p[i] / (z + i)
    t = z + g + 0.5
    return 0.5 * math.log(2 * math.pi) + (z + 0.5) * math.log(t) - t + math.log(x)


def _chi2_pvalue(chi2: float, df: int) -> float:
    """Survival function (upper tail) of the chi-square distribution."""
    if chi2 <= 0:
        return 1.0
    a = df / 2.0
    x = chi2 / 2.0
    # Q(a, x) = 1 - P(a, x)
    p = _regularized_gamma_p(a, x)
    return max(0.0, min(1.0, 1.0 - p))


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def get_verdict(p_value: float) -> tuple:
    """
    Return (verdict_en, verdict_ko) based on p-value thresholds:
      p >= 0.05  → ("conforms", "벤포드 법칙에 부합")
      0.01 <= p < 0.05 → ("suspicious", "이탈 의심")
      p < 0.01   → ("strongly_suspicious", "강한 이탈 의심")
    """
    if p_value >= 0.05:
        return ("conforms", "벤포드 법칙에 부합")
    elif p_value >= 0.01:
        return ("suspicious", "이탈 의심")
    else:
        return ("strongly_suspicious", "강한 이탈 의심")


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------

def analyze(numbers_or_text) -> dict:
    """
    Accept either a list of numbers or a text string.
    Run the full Benford's Law pipeline.

    Returns a dict with keys:
        total, valid_count, first_digits,
        observed_freq, observed_pct, expected_pct,
        chi_square, p_value,
        verdict_en, verdict_ko,
        warning
    """
    if isinstance(numbers_or_text, str):
        numbers = parse_input(numbers_or_text)
    else:
        numbers = preprocess(numbers_or_text)

    total_input = len(numbers_or_text) if not isinstance(numbers_or_text, str) else None
    valid_count = len(numbers)

    first_digits = observed_distribution(numbers)
    expected_probs = benford_expected()

    chi2, p_val, df = chi_square_test(first_digits, valid_count, expected_probs)
    verdict_en, verdict_ko = get_verdict(p_val)

    observed_pct = {
        d: (first_digits[d] / valid_count * 100) if valid_count > 0 else 0.0
        for d in range(1, 10)
    }
    expected_pct = {d: expected_probs[d] * 100 for d in range(1, 10)}

    warning = None
    if valid_count < 30:
        warning = f"유효 데이터 수가 {valid_count}개로 30개 미만입니다. 벤포드 법칙 적용에 주의가 필요합니다."

    return {
        "total": total_input if total_input is not None else valid_count,
        "valid_count": valid_count,
        "first_digits": first_digits,
        "observed_freq": dict(first_digits),
        "observed_pct": observed_pct,
        "expected_pct": expected_pct,
        "chi_square": chi2,
        "p_value": p_val,
        "verdict_en": verdict_en,
        "verdict_ko": verdict_ko,
        "warning": warning,
    }


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_results(result: dict) -> None:
    """Pretty-print analysis results with an ASCII bar chart."""
    sep = "=" * 60

    print(sep)
    print("  벤포드 법칙 분석 결과")
    print(sep)

    if result["warning"]:
        print(f"  [경고] {result['warning']}")
        print()

    print(f"  유효 데이터 수 (n)   : {result['valid_count']}")
    print(f"  카이제곱 통계량 (χ²) : {result['chi_square']:.4f}")
    print(f"  p-값                : {result['p_value']:.4f}")
    print(f"  판정 (EN)           : {result['verdict_en']}")
    print(f"  판정 (KO)           : {result['verdict_ko']}")
    print()

    # Verdict colour / symbol
    v = result["verdict_en"]
    if v == "conforms":
        symbol = "[OK] "
    elif v == "suspicious":
        symbol = "[!]  "
    else:
        symbol = "[!!] "
    print(f"  {symbol}{result['verdict_ko']}")
    print()

    # ASCII bar chart header
    BAR_MAX = 30  # characters for 30 %
    print(f"  {'자릿수':^6} {'관측%':>7} {'기대%':>7}  비교 (█=관측, ░=기대 기준)")
    print("  " + "-" * 56)

    obs_pct = result["observed_pct"]
    exp_pct = result["expected_pct"]

    for d in range(1, 10):
        o = obs_pct[d]
        e = exp_pct[d]
        obs_bar = int(round(o / 30 * BAR_MAX))
        exp_bar = int(round(e / 30 * BAR_MAX))
        # Draw observed (solid) up to min, then marker for difference
        bar = "█" * obs_bar
        ghost = "░" * exp_bar
        print(f"  {d:^6} {o:>6.1f}%  {e:>6.1f}%  |{bar:<{BAR_MAX}}| (기대:{ghost})")

    print(sep)
    print()


# ---------------------------------------------------------------------------
# Demo entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Dataset 1: Natural / Fibonacci-like growth ---
    natural_data = [
        1, 14, 25, 35, 43, 72, 115, 187, 302, 489,
        791, 1280, 2071, 3351, 5422, 8773, 14195, 22968, 37163, 60131,
        97294, 157425, 254719, 412144, 666863, 1079007, 1745870, 2824877,
        4570747, 7395624,
    ]

    # --- Dataset 2: Manipulated / uniform-like ---
    manipulated_data = list(range(50, 100))  # 50..99

    print("\n" + "=" * 60)
    print("  [데이터셋 1] 자연 데이터 (피보나치 유사 성장)")
    print("=" * 60)
    r1 = analyze(natural_data)
    print_results(r1)

    print("\n" + "=" * 60)
    print("  [데이터셋 2] 조작 데이터 (균등 분포 50~99)")
    print("=" * 60)
    r2 = analyze(manipulated_data)
    print_results(r2)
