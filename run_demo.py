
import os, json, yaml
import pandas as pd
from data_io import load_price_data, load_macro_data
from macro_engine import compute_macro_bias
from ict_detectors import detect_liquidity_sweeps, detect_fvg, detect_market_structure_shift
from confluence import confluence_signals
from backtester import run_backtest

def main():
    with open("config.yaml","r") as f:
        cfg = yaml.safe_load(f)

    price = load_price_data(symbol="NQ")
    macro = load_macro_data().reindex(price.index).fillna(method="ffill")

    macro_out = compute_macro_bias(macro, cfg["macro_weights"])
    sweep = detect_liquidity_sweeps(price, lookback=12)
    fvg   = detect_fvg(price)
    mss   = detect_market_structure_shift(price, swing_lookback=7)

    conf = confluence_signals(price, macro_out["macro_bias"], sweep, fvg, mss, cfg["sessions"])

    trades, summary = run_backtest(price, conf,
                                   stop_atr_mult=cfg["backtest"]["stop_atr_mult"],
                                   target_r_multiple=cfg["backtest"]["target_r_multiple"],
                                   risk_per_trade=cfg["backtest"]["risk_per_trade"],
                                   daily_flat=cfg["backtest"]["daily_flat"])

    os.makedirs("outputs", exist_ok=True)
    macro_only = macro_out.drop(columns=["macro_bias"], errors="ignore")
    signals = price.join(conf).join(macro_only)
    signals.to_csv("outputs/signals.csv", index_label="timestamp")
    trades.to_csv("outputs/trades.csv", index=False)
    with open("outputs/summary.json","w") as f:
        json.dump(summary, f, indent=2)

    print("Summary:", summary)

if __name__ == "__main__":
    main()
