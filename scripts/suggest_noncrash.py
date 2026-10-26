"""Candidate cutoff finder that skips the 2008-09 and 2020 market-wide crashes
and any window containing a likely split (daily close ratio <0.6 or >1.6).
Only a pre-filter: the backtest script's own split guard is still the final check."""
import sys
import numpy as np
import pandas as pd

DATA = "data/nse_historical"
LB, FW = 252, 126
EXCLUDE = [("2007-07-01", "2009-12-31"), ("2019-08-01", "2021-12-31")]
DEFAULT = ["TCS", "INFY", "RELIANCE", "HDFCBANK", "ITC", "MARUTI",
           "SUNPHARMA", "ADANIENT", "YESBANK"]
N_DECLINE, N_GROWTH, MIN_MOVE = 4, 3, 0.15


def pick(frame, ascending, k):
    chosen = []
    for idx, row in frame.sort_values("fwd", ascending=ascending).iterrows():
        if all(abs(idx - j) >= FW for j in chosen):
            chosen.append(idx)
        if len(chosen) == k:
            break
    return frame.loc[chosen]


def main():
    tickers = sys.argv[1:] or DEFAULT
    for t in tickers:
        df = pd.read_csv("%s/%s.csv" % (DATA, t), usecols=["Date", "Close"],
                         parse_dates=["Date"]).dropna()
        df = df.sort_values("Date").reset_index(drop=True)
        c = df["Close"]
        ratio = c / c.shift(1)
        jump = ((ratio < 0.6) | (ratio > 1.6)).astype(int).cumsum()
        df["fwd"] = c.shift(-FW) / c - 1
        df["jumps"] = jump.shift(-FW) - jump.shift(LB)
        df["i"] = np.arange(len(df))
        ok = df["fwd"].notna() & (df["i"] >= LB) & (df["jumps"] == 0)
        for a, b in EXCLUDE:
            ok &= ~df["Date"].between(a, b)
        cand = df[ok]
        print("\n=== %s (%d usable days) ===" % (t, len(cand)))
        dec = cand[cand["fwd"] < -MIN_MOVE]
        gro = cand[cand["fwd"] > MIN_MOVE]
        for label, sub, asc, k in (("DECLINE", dec, True, N_DECLINE),
                                   ("GROWTH", gro, False, N_GROWTH)):
            if len(sub) == 0:
                print("  %s: none" % label)
                continue
            for _, r in pick(sub, asc, k).iterrows():
                print("  %-8s %s  fwd=%+.3f" % (label, r["Date"].date(), r["fwd"]))


main()
