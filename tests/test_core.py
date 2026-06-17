"""benford.core 모듈 단위 테스트.

실행: python -m pytest -q   또는   python -m unittest
"""

import math
import unittest

from benford import core
from benford.datasets import (
    benford_following_data,
    fabricated_data,
    uniform_random_data,
)


class FirstDigitTests(unittest.TestCase):
    def test_basic_integers(self):
        self.assertEqual(core.first_digit(4823), 4)
        self.assertEqual(core.first_digit(1), 1)
        self.assertEqual(core.first_digit(900), 9)

    def test_decimals(self):
        self.assertEqual(core.first_digit(0.0071), 7)
        self.assertEqual(core.first_digit(0.3), 3)

    def test_negative(self):
        self.assertEqual(core.first_digit(-38.5), 3)

    def test_zero_and_invalid(self):
        self.assertIsNone(core.first_digit(0))
        self.assertIsNone(core.first_digit("abc"))
        self.assertIsNone(core.first_digit(float("nan")))
        self.assertIsNone(core.first_digit(float("inf")))

    def test_string_numbers(self):
        self.assertEqual(core.first_digit("582"), 5)


class BenfordProbabilityTests(unittest.TestCase):
    def test_known_values(self):
        # 1의 비율은 약 30.1%, 9의 비율은 약 4.6%
        self.assertAlmostEqual(core.benford_probability(1), 0.30103, places=4)
        self.assertAlmostEqual(core.benford_probability(9), 0.04576, places=4)

    def test_probabilities_sum_to_one(self):
        total = sum(core.benford_expected_proportions().values())
        self.assertAlmostEqual(total, 1.0, places=10)

    def test_out_of_range(self):
        with self.assertRaises(ValueError):
            core.benford_probability(0)
        with self.assertRaises(ValueError):
            core.benford_probability(10)


class AnalyzeTests(unittest.TestCase):
    def test_counts_and_proportions(self):
        values = [1, 1, 2, 30, 400]  # 첫자리: 1,1,2,3,4
        counts = core.observed_counts(values)
        self.assertEqual(counts[1], 2)
        self.assertEqual(counts[2], 1)
        self.assertEqual(counts[3], 1)
        self.assertEqual(counts[4], 1)
        self.assertEqual(sum(counts.values()), 5)

    def test_benford_following_data_passes(self):
        result = core.analyze(benford_following_data(2000))
        self.assertTrue(result.follows_benford,
                        msg=f"벤포드 따르는 데이터인데 기각됨: chi2={result.chi_square}")

    def test_uniform_data_fails(self):
        result = core.analyze(uniform_random_data(2000))
        self.assertFalse(result.follows_benford,
                         msg=f"균등 난수인데 벤포드로 판정됨: chi2={result.chi_square}")

    def test_fabricated_data_fails(self):
        result = core.analyze(fabricated_data(500))
        self.assertFalse(result.follows_benford)

    def test_result_summary_runs(self):
        result = core.analyze(benford_following_data(100))
        self.assertIsInstance(result.summary(), str)
        self.assertEqual(result.degrees_of_freedom, 8)

    def test_empty_input(self):
        result = core.analyze([])
        self.assertEqual(result.n, 0)
        self.assertFalse(math.isnan(result.chi_square))


if __name__ == "__main__":
    unittest.main()
