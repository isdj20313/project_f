"""
================================================================
 벤포드 법칙 기반 가짜·조작 데이터 검증 도구  (benford.py)
================================================================
 이 프로그램이 하는 일:
   숫자 데이터의 '맨 앞자리 숫자(1~9)'가
   자연스러운 분포(= 벤포드 법칙)를 따르는지 검사해서,
   데이터가 사람이 조작했을 가능성이 있는지 통계적으로 판단합니다.

 벤포드 법칙: 자연스러운 숫자는 맨 앞자리가 1일 확률이 약 30%로 가장 높고,
             9로 갈수록 점점 낮아진다. ( 공식: P(d) = log10(1 + 1/d) )
"""

import math   # 로그(log10), 내림(floor) 같은 수학 계산을 쓰려고 불러옴
import re     # 글자 속에서 특정 기호(₩, $, 쉼표 등)를 찾아 지우려고 불러옴 (정규표현식)
import sys    # 파이썬 시스템 관련 기능 (지금은 확장용으로 남겨둠)


# ---------------------------------------------------------------------------
#  1) 기본 도구 함수들
# ---------------------------------------------------------------------------

def extract_first_digit(n: float) -> int:
    """숫자 n의 '맨 앞 숫자(1~9)'를 골라낸다.  예) 4521→4,  0.038→3,  900→9"""
    if n <= 0:                                    # 0이나 음수는 맨 앞자리를 정할 수 없으므로
        raise ValueError(f"extract_first_digit expects a positive number, got {n}")  # 오류를 냄
    magnitude = math.floor(math.log10(abs(n)))    # log10으로 '몇 자리 수인지' 구함 (4521 → 3)
    leading = abs(n) / (10 ** magnitude)          # 그만큼 나눠 4.521 처럼 1~10 사이 값으로 만듦
    return int(leading)                           # 정수 부분만 취하면 맨 앞 숫자(4)가 됨


def preprocess(raw_numbers) -> list:
    """
    여러 값이 섞인 목록을 받아서 '분석 가능한 양수'만 골라낸다.
    - 돈 기호(₩ $ € £ ¥), % 기호, 쉼표, 공백을 지움
    - 0, 음수, 숫자가 아닌 값은 버림
    """
    result = []                                   # 걸러낸 깨끗한 숫자를 담을 빈 목록
    for item in raw_numbers:                       # 입력된 값들을 하나씩 꺼내서 검사
        s = str(item).strip()                      # 글자로 바꾸고 양옆 공백 제거
        s = re.sub(r'[₩$€£¥%,\s]', '', s)          # 돈 기호·%·쉼표·공백을 전부 지움
        if not s:                                  # 지우고 나서 빈 글자면
            continue                               # 건너뜀
        try:
            val = float(s)                         # 숫자로 바꿔봄
        except ValueError:                         # 숫자로 못 바꾸면 (글자가 섞였으면)
            continue                               # 그 값은 버리고 다음으로
        if val > 0:                                # 0보다 큰 양수만
            result.append(val)                     # 결과 목록에 추가
    return result                                  # 깨끗한 양수 목록을 돌려줌


def parse_input(text: str) -> list:
    """긴 글(텍스트)을 받아서 숫자들로 쪼갠다. (쉼표·줄바꿈·띄어쓰기로 구분)"""
    tokens = re.split(r'[,\n\s]+', text.strip())   # 쉼표·줄바꿈·공백을 기준으로 잘라 목록으로 만듦
    return preprocess(tokens)                       # 위의 정제 함수로 깨끗하게 걸러서 돌려줌


# ---------------------------------------------------------------------------
#  2) 벤포드 법칙 통계 계산
# ---------------------------------------------------------------------------

def benford_expected() -> dict:
    """벤포드 법칙이 예측하는 '기대 확률'을 계산한다.  공식: P(d) = log10(1 + 1/d)"""
    return {d: math.log10(1 + 1 / d) for d in range(1, 10)}  # 1~9 각 숫자의 이론적 확률을 사전으로 만듦


def observed_distribution(numbers: list) -> dict:
    """실제 데이터에서 맨 앞자리 숫자가 각각 몇 번 나왔는지 센다."""
    counts = {d: 0 for d in range(1, 10)}          # 1~9 칸을 0으로 초기화한 '개수 세는 표'
    for n in numbers:                              # 숫자를 하나씩 꺼내서
        try:
            d = extract_first_digit(n)             # 맨 앞자리를 구하고
            if 1 <= d <= 9:                        # 1~9 범위면
                counts[d] += 1                     # 해당 칸의 개수를 1 늘림
        except (ValueError, ZeroDivisionError):    # 혹시 계산 중 문제가 생기면
            pass                                   # 그 숫자는 그냥 무시
    return counts                                  # 자리별 개수 표를 돌려줌


def chi_square_test(observed_counts: dict, total: int, expected_probs: dict):
    """
    카이제곱 적합도 검정: '실제 분포'가 '벤포드 기대 분포'와 얼마나 다른지를 수치로 잰다.
    돌려주는 값: (카이제곱 통계량, p-value, 자유도=8)
    p-value가 작을수록 → 벤포드와 많이 다름 → 조작 의심.
    """
    digits = list(range(1, 10))                     # 검사할 자리 숫자 1~9
    observed = [observed_counts.get(d, 0) for d in digits]   # 실제 관측된 개수 목록 (O)
    expected = [expected_probs[d] * total for d in digits]   # 기대되는 개수 목록 (E = 확률 × 전체개수)

    try:
        from scipy.stats import chisquare          # scipy 라이브러리가 있으면 그걸 사용 (더 정확/간편)
        chi2_stat, p_value = chisquare(f_obs=observed, f_exp=expected)  # 한 줄로 검정 수행
        return float(chi2_stat), float(p_value), 8 # 결과 반환
    except ImportError:                            # scipy가 없으면
        pass                                       # 아래의 직접 계산 방식으로 넘어감

    # scipy가 없을 때를 대비한 '직접 계산' 방식
    chi2_stat = sum(                               # 카이제곱 공식: Σ (관측−기대)² / 기대
        (o - e) ** 2 / e for o, e in zip(observed, expected) if e > 0
    )
    p_value = _chi2_pvalue(chi2_stat, df=8)        # 위에서 구한 통계량으로 p-value 계산
    return float(chi2_stat), float(p_value), 8     # 결과 반환 (자유도는 9−1=8)


# --- 아래 3개 함수(_로 시작)는 scipy 없이 p-value를 구하기 위한 '수학 보조 도구'입니다 ---
#     (고등학교 범위를 넘는 고급 수식이라, 원리만 이해하고 그대로 써도 됩니다)

def _regularized_gamma_p(a: float, x: float, max_iter: int = 200, tol: float = 1e-12) -> float:
    """불완전 감마함수 P(a,x)를 '급수 전개'로 근사 계산 (p-value 계산의 핵심 부품)."""
    if x < 0:
        raise ValueError("x must be >= 0")
    if x == 0:
        return 0.0
    if x < a + 1:                                  # x가 작을 때는 '급수' 방식이 잘 맞음
        term = 1.0 / a                             # 첫 항
        total = term                               # 합계 초기값
        for n in range(1, max_iter):               # 항을 점점 더해가며
            term *= x / (a + n)                     # 다음 항 계산
            total += term                           # 합계에 더함
            if abs(term) < tol * abs(total):        # 더 더해도 거의 안 변하면
                break                               # 멈춤 (충분히 정확)
        return math.exp(-x + a * math.log(x) - _log_gamma(a)) * total  # 최종 값 반환
    else:                                          # x가 클 때는 '연분수' 방식이 더 정확
        return 1.0 - _regularized_gamma_q_cf(a, x, max_iter, tol)


def _regularized_gamma_q_cf(a: float, x: float, max_iter: int = 200, tol: float = 1e-12) -> float:
    """불완전 감마함수 Q(a,x)를 '연분수(Lentz 방법)'로 근사 계산."""
    fpmin = 1e-300                                 # 0으로 나누는 것을 막는 아주 작은 값
    b = x + 1.0 - a                                # 연분수 계산용 초기값들
    c = 1.0 / fpmin
    d = 1.0 / b
    h = d
    for i in range(1, max_iter + 1):               # 정해진 횟수만큼 반복하며 정밀도를 높임
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
        if abs(delta - 1.0) < tol:                 # 값이 거의 안 변하면
            break                                  # 멈춤
    return math.exp(-x + a * math.log(x) - _log_gamma(a)) * h


def _log_gamma(z: float) -> float:
    """감마함수에 로그를 씌운 값 log(Γ(z))을 'Lanczos 근사'로 계산 (계승의 일반화)."""
    g = 7                                          # 근사식에 쓰는 상수
    p = [                                          # 미리 정해진 근사 계수들
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
    if z < 0.5:                                    # z가 작을 때는 반사 공식을 사용
        return math.log(math.pi / math.sin(math.pi * z)) - _log_gamma(1 - z)
    z -= 1
    x = p[0]
    for i in range(1, g + 2):                      # 계수들을 차례로 더해 근사값을 만듦
        x += p[i] / (z + i)
    t = z + g + 0.5
    return 0.5 * math.log(2 * math.pi) + (z + 0.5) * math.log(t) - t + math.log(x)


def _chi2_pvalue(chi2: float, df: int) -> float:
    """카이제곱 값과 자유도로부터 p-value(우연히 이 정도 차이가 날 확률)를 구함."""
    if chi2 <= 0:                                  # 차이가 0이면
        return 1.0                                 # 완전히 일치 → p-value는 1
    a = df / 2.0                                   # 감마함수에 넣을 값 준비
    x = chi2 / 2.0
    p = _regularized_gamma_p(a, x)                 # 누적확률 계산
    return max(0.0, min(1.0, 1.0 - p))             # p-value = 1 - 누적확률 (0~1 범위로 보정)


# ---------------------------------------------------------------------------
#  3) 판정 (p-value를 보고 단계 나누기)
# ---------------------------------------------------------------------------

def get_verdict(p_value: float) -> tuple:
    """p-value 크기에 따라 3단계 판정을 돌려준다. (영어 코드, 한국어 설명)"""
    if p_value >= 0.05:                            # 5% 이상이면 → 차이가 별로 없음
        return ("conforms", "벤포드 법칙에 부합")          # 자연스러운 데이터
    elif p_value >= 0.01:                          # 1%~5% 사이면 → 어느 정도 벗어남
        return ("suspicious", "이탈 의심")               # 주의가 필요
    else:                                          # 1% 미만이면 → 많이 벗어남
        return ("strongly_suspicious", "강한 이탈 의심")   # 조작 가능성 검토 신호


# ---------------------------------------------------------------------------
#  4) 전체 분석 파이프라인 (위 함수들을 순서대로 묶어 실행)
# ---------------------------------------------------------------------------

def analyze(numbers_or_text) -> dict:
    """
    숫자 목록 또는 글(텍스트)을 받아서 전체 분석을 한 번에 수행한다.
    결과를 사전(dict) 형태로 정리해서 돌려준다.
    """
    if isinstance(numbers_or_text, str):           # 입력이 '글'이면
        numbers = parse_input(numbers_or_text)     # 글을 숫자로 쪼개서 정제
    else:                                          # 입력이 '목록'이면
        numbers = preprocess(numbers_or_text)      # 목록을 바로 정제

    total_input = len(numbers_or_text) if not isinstance(numbers_or_text, str) else None  # 원래 입력 개수
    valid_count = len(numbers)                     # 걸러진 유효 숫자 개수 (= 분석에 쓰는 n)

    first_digits = observed_distribution(numbers)  # 실제 자리별 개수 세기
    expected_probs = benford_expected()            # 벤포드 기대 확률 계산

    chi2, p_val, df = chi_square_test(first_digits, valid_count, expected_probs)  # 카이제곱 검정
    verdict_en, verdict_ko = get_verdict(p_val)    # p-value로 판정

    observed_pct = {                               # 자리별 '실제 비율(%)' 계산
        d: (first_digits[d] / valid_count * 100) if valid_count > 0 else 0.0
        for d in range(1, 10)
    }
    expected_pct = {d: expected_probs[d] * 100 for d in range(1, 10)}  # 자리별 '기대 비율(%)'

    warning = None                                 # 경고 메시지 (기본은 없음)
    if valid_count < 30:                           # 데이터가 30개 미만이면
        warning = f"유효 데이터 수가 {valid_count}개로 30개 미만입니다. 벤포드 법칙 적용에 주의가 필요합니다."  # 신뢰도 경고

    return {                                       # 모든 결과를 하나의 사전으로 묶어서 반환
        "total": total_input if total_input is not None else valid_count,  # 전체 입력 개수
        "valid_count": valid_count,                # 유효 숫자 개수
        "first_digits": first_digits,              # 자리별 개수
        "observed_freq": dict(first_digits),       # 자리별 개수(복사본)
        "observed_pct": observed_pct,              # 자리별 실제 비율
        "expected_pct": expected_pct,              # 자리별 기대 비율
        "chi_square": chi2,                        # 카이제곱 통계량
        "p_value": p_val,                          # p-value
        "verdict_en": verdict_en,                  # 판정(영어)
        "verdict_ko": verdict_ko,                  # 판정(한국어)
        "warning": warning,                        # 경고 메시지
    }


# ---------------------------------------------------------------------------
#  5) 결과를 보기 좋게 화면에 출력 (글자로 그린 막대그래프 포함)
# ---------------------------------------------------------------------------

def print_results(result: dict) -> None:
    """analyze()가 만든 결과를 사람이 읽기 좋게 터미널에 출력한다."""
    sep = "=" * 60                                 # 구분선 (= 60개)

    print(sep)
    print("  벤포드 법칙 분석 결과")
    print(sep)

    if result["warning"]:                          # 경고가 있으면
        print(f"  [경고] {result['warning']}")      # 먼저 보여줌
        print()

    print(f"  유효 데이터 수 (n)   : {result['valid_count']}")        # 분석에 쓴 숫자 개수
    print(f"  카이제곱 통계량 (χ²) : {result['chi_square']:.4f}")     # 차이의 크기
    print(f"  p-값                : {result['p_value']:.4f}")        # 우연일 확률
    print(f"  판정 (EN)           : {result['verdict_en']}")         # 판정(영어)
    print(f"  판정 (KO)           : {result['verdict_ko']}")         # 판정(한국어)
    print()

    v = result["verdict_en"]                        # 판정에 따라 표시할 기호 선택
    if v == "conforms":
        symbol = "[OK] "                            # 부합 → OK
    elif v == "suspicious":
        symbol = "[!]  "                            # 의심 → !
    else:
        symbol = "[!!] "                            # 강한 의심 → !!
    print(f"  {symbol}{result['verdict_ko']}")
    print()

    # 글자로 막대그래프 그리기
    BAR_MAX = 30                                    # 30%일 때 막대 길이를 30칸으로 잡음
    print(f"  {'자릿수':^6} {'관측%':>7} {'기대%':>7}  비교 (█=관측, ░=기대 기준)")  # 표 머리글
    print("  " + "-" * 56)

    obs_pct = result["observed_pct"]                # 실제 비율
    exp_pct = result["expected_pct"]                # 기대 비율

    for d in range(1, 10):                          # 1~9 자리마다 한 줄씩 출력
        o = obs_pct[d]                              # 이 자리의 실제 비율
        e = exp_pct[d]                              # 이 자리의 기대 비율
        obs_bar = int(round(o / 30 * BAR_MAX))      # 실제 비율을 막대 칸 수로 변환
        exp_bar = int(round(e / 30 * BAR_MAX))      # 기대 비율을 막대 칸 수로 변환
        bar = "█" * obs_bar                         # 실제 막대(꽉 찬 블록)
        ghost = "░" * exp_bar                       # 기대 막대(옅은 블록)
        print(f"  {d:^6} {o:>6.1f}%  {e:>6.1f}%  |{bar:<{BAR_MAX}}| (기대:{ghost})")  # 한 줄 출력

    print(sep)
    print()


# ---------------------------------------------------------------------------
#  6) 프로그램을 직접 실행했을 때 동작하는 '데모'
# ---------------------------------------------------------------------------

if __name__ == "__main__":                          # 이 파일을 직접 실행할 때만 아래가 동작 (import 시엔 동작 안 함)
    # --- 데이터셋 1: 자연스러운 데이터 (점점 커지는 피보나치 비슷한 수들) ---
    natural_data = [
        1, 14, 25, 35, 43, 72, 115, 187, 302, 489,
        791, 1280, 2071, 3351, 5422, 8773, 14195, 22968, 37163, 60131,
        97294, 157425, 254719, 412144, 666863, 1079007, 1745870, 2824877,
        4570747, 7395624,
    ]

    # --- 데이터셋 2: 조작된 느낌의 데이터 (50~99까지 고르게) ---
    manipulated_data = list(range(50, 100))         # 50부터 99까지의 숫자 목록

    print("\n" + "=" * 60)
    print("  [데이터셋 1] 자연 데이터 (피보나치 유사 성장)")
    print("=" * 60)
    r1 = analyze(natural_data)                       # 자연 데이터 분석
    print_results(r1)                                # 결과 출력

    print("\n" + "=" * 60)
    print("  [데이터셋 2] 조작 데이터 (균등 분포 50~99)")
    print("=" * 60)
    r2 = analyze(manipulated_data)                   # 조작 데이터 분석
    print_results(r2)                                # 결과 출력
