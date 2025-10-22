
import pandas as pd
import numpy as np

def _synthetic_price(start="2025-09-01 06:00", periods=6*24*5, freq="5T", seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, periods=periods, freq=freq)
    price = 100 + np.cumsum(rng.normal(0, 0.1, size=len(idx)))
    wiggle = np.sin(np.linspace(0, 12*np.pi, len(idx))) * 0.3
    mid = price + wiggle
    o = mid + rng.normal(0, 0.05, len(idx))
    h = o + np.abs(rng.normal(0.08, 0.04, len(idx)))
    l = o - np.abs(rng.normal(0.08, 0.04, len(idx)))
    c = o + rng.normal(0, 0.08, len(idx))
    df = pd.DataFrame({"open":o, "high":h, "low":l, "close":c}, index=idx)
    df.index.name = "timestamp"
    return df

def _synthetic_macro(start="2025-09-01 06:00", periods=6*24*5, freq="5T", seed=7):
    idx = pd.date_range(start=start, periods=periods, freq=freq)
    rng = np.random.default_rng(seed)
    dxy = 103 + np.cumsum(rng.normal(0, 0.02, len(idx)))
    yld = 4.0 + np.cumsum(rng.normal(0, 0.001, len(idx)))
    vix = 16 + np.cumsum(rng.normal(0, 0.03, len(idx)))
    gamma = np.clip(0.5 + np.sin(np.linspace(0, 4*np.pi, len(idx))) * 0.3 + rng.normal(0, 0.02, len(idx)), 0, 1)
    df = pd.DataFrame({"dxy": dxy, "yield": yld, "vix": vix, "gamma": gamma}, index=idx)
    df.index.name = "timestamp"
    return df

def load_price_data(symbol: str) -> pd.DataFrame:
    return _synthetic_price()

def load_macro_data() -> pd.DataFrame:
    return _synthetic_macro()
