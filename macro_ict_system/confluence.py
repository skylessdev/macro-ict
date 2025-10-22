
import pandas as pd
from utils import in_session

def confluence_signals(price_df: pd.DataFrame,
                       macro_bias: pd.Series,
                       sweep_sig: pd.Series,
                       fvg_sig: pd.Series,
                       mss_sig: pd.Series,
                       sessions: dict) -> pd.DataFrame:
    idx = price_df.index
    sess_mask = pd.Series(False, index=idx)
    for name, win in sessions.items():
        start, end = win["start"], win["end"]
        sess_mask |= pd.Series([in_session(ts, start, end) for ts in idx], index=idx)

    long_ok  = (macro_bias==1) & ((sweep_sig==1) | (fvg_sig==1)) & (mss_sig==1) & sess_mask
    short_ok = (macro_bias==-1)& ((sweep_sig==-1)| (fvg_sig==-1))& (mss_sig==-1)& sess_mask

    signal = pd.Series(0, index=idx, dtype=int)
    signal[long_ok] = 1; signal[short_ok] = -1

    df = pd.DataFrame({
        "macro_bias": macro_bias.reindex(idx).fillna(0).astype(int),
        "sweep": sweep_sig.reindex(idx).fillna(0).astype(int),
        "fvg":   fvg_sig.reindex(idx).fillna(0).astype(int),
        "mss":   mss_sig.reindex(idx).fillna(0).astype(int),
        "session": sess_mask.astype(int),
        "signal": signal
    }, index=idx)
    return df
