
import pandas as pd
import numpy as np

def in_session(ts: pd.Timestamp, start_str: str, end_str: str) -> bool:
    s = pd.to_datetime(start_str).time()
    e = pd.to_datetime(end_str).time()
    t = ts.time()
    if s <= e:
        return s <= t <= e
    else:
        return not (e < t < s)

def compute_atr(df: pd.DataFrame, period:int=14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = (h - l).abs()
    tr = pd.concat([tr, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    atr.name = "atr"
    return atr
