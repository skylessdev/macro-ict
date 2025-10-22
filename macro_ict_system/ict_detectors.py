
import pandas as pd
import numpy as np

def detect_liquidity_sweeps(df: pd.DataFrame, lookback:int=10) -> pd.Series:
    h = df["high"]; l = df["low"]; o = df["open"]; c = df["close"]
    prior_max = h.shift(1).rolling(lookback).max()
    prior_min = l.shift(1).rolling(lookback).min()
    bull = (h > prior_max) & (c <= prior_max) & (c > o)
    bear = (l < prior_min) & (c >= prior_min) & (c < o)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[bull] = 1; sig[bear] = -1
    sig.name = "sweep_signal"
    return sig

def detect_fvg(df: pd.DataFrame) -> pd.Series:
    h = df["high"]; l = df["low"]
    bull = l > h.shift(2)
    bear = h < l.shift(2)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[bull] = 1; sig[bear] = -1
    sig.name = "fvg_signal"
    return sig

def detect_market_structure_shift(df: pd.DataFrame, swing_lookback:int=5) -> pd.Series:
    h = df["high"]; l = df["low"]
    swing_high = (h == h.rolling(swing_lookback, center=True).max())
    swing_low  = (l == l.rolling(swing_lookback, center=True).min())
    state = pd.Series(0, index=df.index, dtype=int)
    last_high = None; last_low = None; trend = 0
    for i in range(len(df)):
        if swing_high.iloc[i]:
            if last_high is not None and h.iloc[i] > last_high:
                trend = 1
            last_high = h.iloc[i]
        if swing_low.iloc[i]:
            if last_low is not None and l.iloc[i] < last_low:
                trend = -1
            last_low = l.iloc[i]
        state.iloc[i] = trend
    mss = state.diff().fillna(0)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[mss > 0] = 1; sig[mss < 0] = -1
    sig.name = "mss_signal"
    return sig
