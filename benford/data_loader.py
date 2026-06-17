import numpy as np
import pandas as pd


def load_csv(filepath):
    df = pd.read_csv(filepath)
    numbers = []
    for col in df.columns:
        numeric = pd.to_numeric(df[col], errors="coerce").dropna()
        numbers.extend(numeric.tolist())
    return numbers


def generate_population_data(n=500):
    rng = np.random.default_rng(42)
    return rng.lognormal(mean=10, sigma=2, size=n).tolist()


def generate_fake_data(n=500):
    rng = np.random.default_rng(42)
    return rng.integers(10, 1000, size=n).tolist()


def generate_fraud_data(n=500):
    rng = np.random.default_rng(42)
    leading = rng.choice([4, 5, 6, 7], size=n, p=[0.28, 0.27, 0.25, 0.20])
    result = []
    for d in leading:
        magnitude = rng.integers(1, 100)
        result.append(d * magnitude)
    return result
