
import pandas as pd
import numpy as np

def zscore(s: pd.Series, window: int = 60) -> pd.Series:
    z = (s - s.rolling(window).mean()) / (s.rolling(window).std().replace(0, np.nan))
    return z

def compute_macro_bias(macro_df: pd.DataFrame, weights: dict, z_window:int=60) -> pd.DataFrame:
    """
    Compute weighted macro score and bias.
    Expects columns: dxy, yield, vix, gamma
    We derive features: dxy_chg, yield_chg, vix_chg, gamma_level
    weights example: {'dxy_chg': -1.0, 'yield_chg': -0.8, 'vix_chg': -0.7, 'gamma_level': 0.6}
    Returns DataFrame with columns: macro_score, macro_bias in {-1, 0, +1}
    """
    df = macro_df.copy()
    df["dxy_chg"]   = df["dxy"].diff()
    df["yield_chg"] = df["yield"].diff()
    df["vix_chg"]   = df["vix"].diff()
    df["gamma_level"] = df["gamma"]

    feats = {}
    for k in ["dxy_chg","yield_chg","vix_chg","gamma_level"]:
        feats[k] = zscore(df[k], window=z_window).fillna(0.0)

    w = pd.Series(weights, dtype=float)
    if w.abs().sum() == 0:
        w = pd.Series({k:1.0 for k in feats})
    w = w / w.abs().sum()

    score = sum(w[k] * feats[k] for k in feats)
    score = score.rename("macro_score")

    # dynamic threshold using rolling volatility of score
    deadzone = score.rolling(120).std()
    if float(deadzone.dropna().mean() if deadzone.dropna().size else 0) == 0:
        dz = 0.25
    else:
        dz = 0.25 * float(deadzone.dropna().mean())
    bias = score.apply(lambda x: 1 if x>dz else (-1 if x<-dz else 0)).rename("macro_bias")

    out = pd.concat([score, bias], axis=1)
    return out
