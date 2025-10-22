
import pandas as pd
import numpy as np
from utils import compute_atr

def run_backtest(price_df: pd.DataFrame, signal_df: pd.DataFrame,
                 stop_atr_mult: float=0.7, target_r_multiple: float=2.0,
                 risk_per_trade: float=1.0, daily_flat: bool=True):
    df = price_df.copy()
    df["signal"] = signal_df["signal"].reindex(df.index).fillna(0)
    df["atr"] = compute_atr(df, 14).bfill()

    trades = []
    position = 0
    entry_px = np.nan
    stop_px = np.nan
    target_px = np.nan
    entry_time = None
    prev_sig = 0
    prev_day = None

    def close_trade(ts, px, reason):
        nonlocal position, entry_px, stop_px, target_px, entry_time
        if position == 0 or pd.isna(entry_px):
            return
        r = (px - entry_px) if position>0 else (entry_px - px)
        denom = abs(entry_px - stop_px) if abs(entry_px - stop_px)>0 else np.nan
        rR = r / denom if denom==denom else 0.0
        trades.append({
            "entry_time": entry_time, "exit_time": ts,
            "side": "LONG" if position>0 else "SHORT",
            "entry": entry_px, "exit": px, "R": rR, "reason": reason
        })
        position = 0; entry_px = stop_px = target_px = np.nan; entry_time = None

    for ts in df.index:
        o,h,l,c = df.loc[ts, ["open","high","low","close"]]
        sig = int(df.loc[ts, "signal"])
        atr = float(df.loc[ts, "atr"])
        day = ts.date()

        if daily_flat and prev_day is not None and day != prev_day and position!=0:
            close_trade(ts, c, "daily_flat")

        if position == 0 and prev_sig == 0 and sig != 0:
            position = sig
            entry_px = o
            stop_dist = atr * stop_atr_mult
            stop_px = entry_px - stop_dist if position>0 else entry_px + stop_dist
            target_px = entry_px + target_r_multiple * stop_dist if position>0 else entry_px - target_r_multiple * stop_dist
            entry_time = ts

        if position != 0:
            if position > 0:
                if l <= stop_px:
                    close_trade(ts, stop_px, "stop")
                elif h >= target_px:
                    close_trade(ts, target_px, "target")
            else:
                if h >= stop_px:
                    close_trade(ts, stop_px, "stop")
                elif l <= target_px:
                    close_trade(ts, target_px, "target")

        prev_sig = sig
        prev_day = day

    if position != 0:
        close_trade(df.index[-1], df["close"].iloc[-1], "final_close")

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        summary = {
            "trades": int(len(trades_df)),
            "win_rate": float((trades_df["R"]>0).mean()),
            "avg_R": float(trades_df["R"].mean()),
            "total_R": float(trades_df["R"].sum()),
            "max_R": float(trades_df["R"].max()),
            "min_R": float(trades_df["R"].min()),
        }
    else:
        summary = {"trades": 0, "win_rate": None, "avg_R": None, "total_R": 0.0}
    return trades_df, summary
