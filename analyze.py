#!/usr/bin/env python3
"""벤포드 법칙 분석 명령줄(CLI) 도구.

사용 예
-------
1) 내장 예시 데이터로 분석:
       python analyze.py --sample "한국 도시 인구"

2) 사용할 수 있는 예시 데이터 이름 보기:
       python analyze.py --list

3) CSV 파일의 특정 열을 분석:
       python analyze.py --csv data/numbers.csv --column 매출액

4) 한 줄에 숫자 하나씩 들어 있는 텍스트 파일을 분석:
       python analyze.py --file data/numbers.txt
"""

from __future__ import annotations

import argparse
import csv
import sys

from benford import analyze
from benford.core import DIGITS
from benford.datasets import get_named_datasets


def _read_text_file(path: str) -> list[float]:
    values: list[float] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip().replace(",", "")
            if not line:
                continue
            try:
                values.append(float(line))
            except ValueError:
                continue
    return values


def _read_csv_column(path: str, column: str | None) -> list[float]:
    values: list[float] = []
    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return values
        if column is None:
            column = reader.fieldnames[0]
        if column not in reader.fieldnames:
            raise SystemExit(
                f"'{column}' 열을 찾을 수 없습니다. 사용 가능한 열: {reader.fieldnames}"
            )
        for row in reader:
            raw = (row.get(column) or "").strip().replace(",", "")
            if not raw:
                continue
            try:
                values.append(float(raw))
            except ValueError:
                continue
    return values


def _print_report(values: list[float], title: str) -> None:
    result = analyze(values)

    print("=" * 56)
    print(f" 벤포드 법칙 분석: {title}")
    print("=" * 56)
    print(f" 분석 숫자 개수: {result.n}")
    print()
    print(" 자리 |  관측수 |  관측%  |  벤포드% |  차이")
    print(" -----+---------+---------+----------+--------")
    for d in DIGITS:
        obs_p = result.observed_proportions[d] * 100
        exp_p = result.expected_proportions[d] * 100
        diff = obs_p - exp_p
        bar = "#" * int(round(obs_p / 2))
        print(
            f"  {d}  | {result.observed_counts[d]:>7} | {obs_p:>6.2f}% | "
            f"{exp_p:>7.2f}% | {diff:>+6.2f}  {bar}"
        )
    print()
    print(f" 카이제곱 통계량 : {result.chi_square:.3f}")
    print(f" 자유도          : {result.degrees_of_freedom}")
    print(f" 임계값(5%)      : {result.critical_value:.3f}")
    if result.p_value is not None:
        print(f" p-value         : {result.p_value:.4f}")
    print(f" MAD             : {result.mad:.4f}")
    print()
    print(f" >> 판정: {result.verdict}")
    print("=" * 56)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="벤포드 법칙으로 숫자 데이터의 첫자리 분포를 분석합니다."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--sample", help="내장 예시 데이터 이름")
    group.add_argument("--csv", help="분석할 CSV 파일 경로")
    group.add_argument("--file", help="한 줄에 숫자 하나씩 든 텍스트 파일 경로")
    group.add_argument("--list", action="store_true", help="내장 예시 데이터 목록 표시")
    parser.add_argument("--column", help="CSV에서 분석할 열 이름(기본: 첫 번째 열)")

    args = parser.parse_args(argv)
    datasets = get_named_datasets()

    if args.list or (not args.sample and not args.csv and not args.file):
        print("사용 가능한 내장 예시 데이터:")
        for name in datasets:
            print(f"  - {name}")
        if not args.list:
            print("\n예: python analyze.py --sample \"한국 도시 인구\"")
        return 0

    if args.sample:
        if args.sample not in datasets:
            print(f"'{args.sample}' 예시 데이터가 없습니다. --list 로 목록을 확인하세요.",
                  file=sys.stderr)
            return 1
        _print_report(datasets[args.sample], args.sample)
        return 0

    if args.csv:
        values = _read_csv_column(args.csv, args.column)
        _print_report(values, f"{args.csv} ({args.column or '첫 열'})")
        return 0

    if args.file:
        values = _read_text_file(args.file)
        _print_report(values, args.file)
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
