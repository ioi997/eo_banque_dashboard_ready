#!/usr/bin/env python3
"""
Walk-forward optimisation of the v2 strategy parameters.

1. TRAIN: try every parameter combination on days 365 to 181 ago.
2. TEST : judge the best combinations on the last 180 days, which the tuning never saw.
Picking on the train window and judging on the test window avoids "fitting the past".

Signals are vectorised (all indicators use only past data), so the grid runs in seconds.
Usage: python bot/optimize.py
Not financial advice.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import main  # noqa: E402

ROOT = main.ROOT
COST_BPS = 25

GRID = {
    "guard_ma": [20, 50, 100],     # no buy while price is below this moving average
    "crash_drop": [5, 10, 15],     # no buy after falling more than X% in 10 days
    "step": [1, 3, 7],             # review positions every N days
    "regime_ma": [0, 50, 100],     # crypto switch: BTC above this average (0 = switch off)
}
DEFAULT = {"guard_ma": 50, "crash_drop": 10, "step": 7, "regime_ma": 50}  # current v2


def base_frames(df: pd.DataFrame) -> dict[str, pd.Series]:
    c = df["Close"]
    r = main.rsi(c)
    h = main.macd_hist(c)
    sma50, sma200 = c.rolling(50).mean(), c.rolling(200).mean()
    score = np.where(c > sma200, 1, -1) + np.where(sma50 > sma200, 1, -1)
    score = score + np.where((h > 0) & (h > h.shift()), 1, np.where((h < 0) & (h < h.shift()), -1, 0))
    cross_up = ((r.shift() < 30) & (r >= 30)).astype(int)
    bounced = cross_up.rolling(3, min_periods=1).max().astype(bool)
    rsi_pts = np.where(bounced, 2, np.where(r < 30, 0, np.where(r > 80, -2, np.where(r > 70, -1, 0))))
    return {
        "close": c, "rsi": r,
        "score": pd.Series(score + rsi_pts, index=c.index),
        "ret10": (c / c.shift(10) - 1) * 100,
        "valid": sma200.notna(),
        **{f"ma{n}": c.rolling(n).mean() for n in GRID["guard_ma"]},
    }


def buy_flags(frames: dict, is_crypto: bool, regime: pd.Series | None, p: dict) -> pd.Series:
    f = frames
    ok = f["valid"] & (f["score"] >= 1) & (f["rsi"] <= 75)
    ok &= f["close"] >= f[f"ma{p['guard_ma']}"]
    ok &= ~(f["ret10"] < -p["crash_drop"])
    if is_crypto and regime is not None:
        ok &= regime.reindex(ok.index).fillna(True).astype(bool)
    return ok


def simulate(buys: pd.DataFrame, rets: pd.DataFrame, days: pd.DatetimeIndex, step: int | None):
    """buys: bool frame (decision day x asset). step None = buy and hold."""
    names = list(rets.columns)
    eq, prev_w, trades = [1000.0], np.zeros(len(names)), 0
    pos = {d: i for i, d in enumerate(rets.index)}
    for i, d_prev in enumerate(days):
        d = rets.index[pos[d_prev] + 1]
        if step is None:
            w = np.full(len(names), 1 / len(names))
        elif i % step == 0:
            b = buys.loc[d_prev].to_numpy(dtype=bool)
            w = b / b.sum() if b.any() else np.zeros(len(names))
        else:
            w = prev_w
        cost = 0.0 if step is None else np.abs(w - prev_w).sum() * COST_BPS / 10_000
        trades += 0 if step is None else int(((w > 0) != (prev_w > 0)).sum())
        eq.append(eq[-1] * (1 + float(w @ rets.loc[d].to_numpy()) - cost))
        prev_w = w
    s = pd.Series(eq)
    return {"ret": (eq[-1] / 1000 - 1) * 100, "dd": float((s / s.cummax() - 1).min() * 100), "trades": trades}


def run(cfg: dict, train_days: int, test_days: int) -> str:
    settings = cfg.get("settings", {})
    hist, kind = {}, {}
    for a in cfg["assets"]:
        df = main.get_history(a)
        if df is not None and len(df) > 260:
            hist[a["name"]], kind[a["name"]] = df, a["type"]
    regime_name = next((a["name"] for a in cfg["assets"]
                        if a.get("yahoo") == settings.get("regime_asset", "BTC-USD")), None)

    closes = pd.DataFrame({n: d["Close"] for n, d in hist.items()}).sort_index().ffill()
    rets = closes.pct_change().fillna(0)
    frames = {n: base_frames(d) for n, d in hist.items()}
    btc = hist[regime_name]["Close"] if regime_name in hist else None

    # Decision days need 200-day history for every asset
    first_ok = max(f["valid"].idxmax() for f in frames.values())
    usable = closes.index[(closes.index >= first_ok)][:-1]
    total = min(train_days + test_days, len(usable))
    test_days = min(test_days, total // 2)
    train_days = total - test_days
    train, test = usable[-total:-test_days], usable[-test_days:]

    def buys_for(p: dict) -> pd.DataFrame:
        regime = None
        if p["regime_ma"] and btc is not None:
            regime = btc > btc.rolling(p["regime_ma"]).mean()
        cols = {}
        for n, f in frames.items():
            fl = buy_flags(f, kind[n] == "crypto", regime, p)
            cols[n] = fl.reindex(closes.index).ffill().fillna(False)
        return pd.DataFrame(cols)[list(closes.columns)]

    rows = []
    for combo in itertools.product(*GRID.values()):
        p = dict(zip(GRID, combo))
        b = buys_for(p)
        tr = simulate(b, rets, train, p["step"])
        te = simulate(b, rets, test, p["step"])
        rows.append({**p, "train_ret": tr["ret"], "train_dd": tr["dd"],
                     "test_ret": te["ret"], "test_dd": te["dd"], "test_trades": te["trades"]})
    res = pd.DataFrame(rows)
    # Rank on TRAIN only: return, then smaller drawdown
    res = res.sort_values(["train_ret", "train_dd"], ascending=[False, False]).reset_index(drop=True)

    bh_tr = simulate(None, rets, train, None)
    bh_te = simulate(None, rets, test, None)
    d = res[(res.guard_ma == DEFAULT["guard_ma"]) & (res.crash_drop == DEFAULT["crash_drop"])
            & (res.step == DEFAULT["step"]) & (res.regime_ma == DEFAULT["regime_ma"])].iloc[0]
    best = res.iloc[0]
    test_rank = res["test_ret"].rank(ascending=False).astype(int)

    def ptxt(r) -> str:
        reg = f"BTC>{int(r.regime_ma)}d" if r.regime_ma else "off"
        return f"MA {int(r.guard_ma)} / drop {int(r.crash_drop)}% / every {int(r.step)}d / switch {reg}"

    improved = best.test_ret > d.test_ret + 1
    verdict = (f"✅ Tuned settings beat current v2 on unseen data by {best.test_ret - d.test_ret:.1f} points. "
               "Worth switching." if improved else
               "➖ Tuned settings do **not** clearly beat current v2 on unseen data. Keep current v2: "
               "the extra tuning was fitting noise.")

    tr_rng = f"{train[0].date()} to {train[-1].date()}"
    te_rng = f"{test[0].date()} to {test[-1].date()}"
    L = ["# ⚙️ Optimisation: walk-forward test", "", verdict, "",
         f"- Train window (used to pick): {tr_rng} ({len(train)} days)",
         f"- Test window (unseen, used to judge): {te_rng} ({len(test)} days)",
         f"- Combinations tried: {len(res)}", "",
         "## Head to head", "",
         "| Strategy | Settings | Train return | Test return | Test worst drop | Test trades |",
         "|---|---|---:|---:|---:|---:|",
         f"| Best on train | {ptxt(best)} | {main.fmt_pct(best.train_ret)} | **{main.fmt_pct(best.test_ret)}** | "
         f"{best.test_dd:.1f}% | {int(best.test_trades)} |",
         f"| Current v2 | {ptxt(d)} | {main.fmt_pct(d.train_ret)} | **{main.fmt_pct(d.test_ret)}** | "
         f"{d.test_dd:.1f}% | {int(d.test_trades)} |",
         f"| Buy and hold | all assets, equal weight | {main.fmt_pct(bh_tr['ret'])} | **{main.fmt_pct(bh_te['ret'])}** | "
         f"{bh_te['dd']:.1f}% | 0 |", "",
         "## Top 10 on train, and how they did on unseen data", "",
         "| Rank | Settings | Train return | Test return | Test rank (of %d) |" % len(res),
         "|---:|---|---:|---:|---:|"]
    for i, r in res.head(10).iterrows():
        L.append(f"| {i + 1} | {ptxt(r)} | {main.fmt_pct(r.train_ret)} | {main.fmt_pct(r.test_ret)} | {test_rank[i]} |")
    L.append("")

    L += ["## Which settings matter (average test return)", ""]
    for k in GRID:
        g = res.groupby(k)["test_ret"].mean()
        L.append(f"- **{k}**: " + ", ".join(f"{int(v)} → {main.fmt_pct(x)}" for v, x in g.items()))
    L += ["", "---",
          "_If the top train settings land at random ranks on the test window, the strategy has no stable "
          f"edge from tuning. Fee: {COST_BPS} bps per trade. Not financial advice._"]
    return "\n".join(L).replace(" → ", ": ")


def cli() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-days", type=int, default=185)
    ap.add_argument("--test-days", type=int, default=180)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--out", default=str(ROOT / "optimize.md"))
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    md = run(cfg, args.train_days, args.test_days)
    Path(args.out).write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(cli())
