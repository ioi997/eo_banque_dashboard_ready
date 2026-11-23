#!/usr/bin/env python3
"""
Backtest the tips bot and compare strategy versions over several periods.

For every past day the signals are recomputed using ONLY the data available that day
(no look-ahead). Strategies compared:
  - Bot v1 (daily)  : original rules, rebalanced every day
  - Bot v2 (weekly) : falling-knife guards, crypto switch, RSI bounce, rebalanced every 7 days
  - Buy and hold    : equal weight in the whole watchlist from day 1
Each bot holds, in equal weights, every asset flagged as a buy (score >= 1 and RSI <= 75)
and sits in cash (stablecoins) otherwise. Trading costs are included.

Usage: python bot/backtest.py --periods 30,90,180
Not financial advice.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import main  # noqa: E402

ROOT = main.ROOT

# label -> (rules version, rebalance every N days)
STRATEGIES = {
    "Bot v1 (daily)": ("v1", 1),
    "Bot v2 (weekly)": ("v2", 7),
}


def is_buy(sig: main.Signal) -> bool:
    return sig.score >= 1 and sig.rsi <= 75


def max_drawdown(equity: list[float]) -> float:
    s = pd.Series(equity)
    return float((s / s.cummax() - 1).min() * 100)


def load(cfg: dict, need_days: int):
    min_days = int(cfg.get("settings", {}).get("min_history_days", 210))
    hist, meta = {}, {}
    for a in cfg["assets"]:
        df = main.get_history(a)
        if df is None or len(df) < min_days + 5:
            print(f"skip {a['name']} (no data)", file=sys.stderr)
            continue
        hist[a["name"]], meta[a["name"]] = df, a
    if not hist:
        raise SystemExit("No data downloaded.")
    regime_name = next((a["name"] for a in cfg["assets"]
                        if a.get("yahoo") == cfg.get("settings", {}).get("regime_asset", "BTC-USD")), None)
    return hist, meta, regime_name, min_days


def run(cfg: dict, periods: list[int], cost_bps: float, capital: float) -> str:
    longest = max(periods)
    hist, meta, regime_name, min_days = load(cfg, longest)
    names = list(hist)

    closes = pd.DataFrame({n: df["Close"] for n, df in hist.items()}).sort_index().ffill()
    rets = closes.pct_change().fillna(0)
    idx = closes.index
    longest = min(longest, len(idx) - min_days - 1)
    periods = sorted({min(p, longest) for p in periods})
    decision_days = list(idx[-(longest + 1):-1])

    # Crypto switch per day (Bitcoin vs its own 50-day average, known that day)
    if regime_name:
        btc = closes[regime_name]
        regime = (btc > btc.rolling(50).mean()).to_dict()
    else:
        regime = {}

    # Signals per strategy per decision day
    signals: dict[str, dict] = {v: {} for v, _ in STRATEGIES.values()}
    for d in decision_days:
        for version in signals:
            day = {}
            for n, df in hist.items():
                past = df[df.index <= d]
                if len(past) >= min_days:
                    day[n] = main.analyse(meta[n], past, version, bool(regime.get(d, True)))
            signals[version][d] = day

    def simulate(days: list, version: str | None, step: int):
        eq, prev_w, trades, invested = [capital], pd.Series(0.0, index=names), 0, 0
        for i, d_prev in enumerate(days):
            d = idx[idx.get_loc(d_prev) + 1]
            if version is None:  # buy and hold
                w = pd.Series(1 / len(names), index=names)
            elif i % step == 0:
                picks = [n for n, s in signals[version][d_prev].items() if is_buy(s)]
                w = pd.Series(0.0, index=names)
                if picks:
                    w[picks] = 1 / len(picks)
            else:
                w = prev_w
            cost = 0.0 if version is None else float((w - prev_w).abs().sum()) * cost_bps / 10_000
            if version is not None:
                trades += int(((w > 0) != (prev_w > 0)).sum())
            invested += int(w.sum() > 0)
            eq.append(eq[-1] * (1 + float((w * rets.loc[d, names]).sum()) - cost))
            prev_w = w
        return eq, trades, invested

    # --- Results per period ---------------------------------------------------
    results = {}  # (period, label) -> dict
    for p in periods:
        days = decision_days[-p:]
        for label, (version, step) in STRATEGIES.items():
            eq, tr, inv = simulate(days, version, step)
            results[(p, label)] = dict(ret=(eq[-1] / capital - 1) * 100, dd=max_drawdown(eq),
                                       end=eq[-1], trades=tr, invested=inv, n=len(days))
        eq, _, inv = simulate(days, None, 1)
        results[(p, "Buy and hold")] = dict(ret=(eq[-1] / capital - 1) * 100, dd=max_drawdown(eq),
                                           end=eq[-1], trades=0, invested=inv, n=len(days))
    labels = list(STRATEGIES) + ["Buy and hold"]

    # --- Signal quality over the longest period --------------------------------
    quality = []
    for label, (version, _) in STRATEGIES.items():
        f5s = []
        for d, day in signals[version].items():
            loc = idx.get_loc(d)
            if loc + 5 >= len(idx):
                continue
            for n, s in day.items():
                if is_buy(s):
                    f5s.append((closes[n].iloc[loc + 5] / closes[n].iloc[loc] - 1) * 100)
        f5 = pd.Series(f5s, dtype=float)
        quality.append((label, len(f5), f5.mean() if len(f5) else np.nan,
                        (f5 > 0).mean() * 100 if len(f5) else np.nan))

    # --- Report ------------------------------------------------------------------
    end = idx[-1].date()
    wins = {lab: sum(results[(p, lab)]["ret"] == max(results[(p, l)]["ret"] for l in labels) for p in periods)
            for lab in labels}
    best = max(wins, key=lambda l: (wins[l], np.mean([results[(p, l)]["ret"] for p in periods])))

    L = [f"# 🧪 Backtest: v1 vs v2 vs buy and hold (to {end})", "",
         f"**Winner: {best}**, best return in {wins[best]} of {len(periods)} periods "
         f"({', '.join(f'{p}d' for p in periods)}).", "",
         "## 💰 Return (start 1,000)", "",
         "| Period | " + " | ".join(labels) + " |",
         "|---|" + "---:|" * len(labels)]
    for p in periods:
        cells = []
        top = max(results[(p, l)]["ret"] for l in labels)
        for l in labels:
            r = results[(p, l)]
            txt = f"{main.fmt_pct(r['ret'])} ({r['end']:,.0f})"
            cells.append(f"**{txt}**" if r["ret"] == top else txt)
        L.append(f"| {p} days | " + " | ".join(cells) + " |")
    L.append("")

    L += ["## 📉 Worst drop along the way", "",
          "| Period | " + " | ".join(labels) + " |", "|---|" + "---:|" * len(labels)]
    for p in periods:
        L.append(f"| {p} days | " + " | ".join(f"{results[(p, l)]['dd']:.1f}%" for l in labels) + " |")
    L.append("")

    L += ["## 🔁 Activity", "",
          "| Period | " + " | ".join(f"{l}: trades / days invested" for l in STRATEGIES) + " |",
          "|---|" + "---:|" * len(STRATEGIES)]
    for p in periods:
        L.append(f"| {p} days | " + " | ".join(
            f"{results[(p, l)]['trades']} / {results[(p, l)]['invested']} of {results[(p, l)]['n']}"
            for l in STRATEGIES) + " |")
    L.append("")

    L += [f"## 🎯 Buy signal quality (last {longest} days)", "",
          "| Strategy | Buy signals | Avg move 5 days later | % up 5 days later |", "|---|---:|---:|---:|"]
    for label, n, avg, hit in quality:
        L.append(f"| {label} | {n} | {main.fmt_pct(avg) if n else 'n/a'} | "
                 f"{f'{hit:.0f}%' if n else 'n/a'} |")
    L.append("")

    last = decision_days[-1]
    L += ["## 📌 What v2 says now", ""]
    now = signals["v2"][last]
    buys = [n for n, s in now.items() if is_buy(s)]
    L.append(f"- Buy list: {', '.join(buys) if buys else 'none, stay in cash / stablecoins'}")
    L.append(f"- Crypto switch: {'ON' if regime.get(last, True) else 'OFF (Bitcoin below its 50-day average)'}")
    L += ["", "---",
          "_v2 rules: no buy when price is below its 50-day average or down more than 10% in 2 weeks; "
          "no crypto buys while Bitcoin is below its 50-day average; RSI adds +2 only when it climbs back "
          "above 30; positions reviewed every 7 days. Fee: "
          f"{cost_bps:.0f} bps per trade. ETFs are flat on weekends. Not financial advice._"]
    return "\n".join(L)


def cli() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--periods", default="30,90,180")
    ap.add_argument("--cost-bps", type=float, default=25, help="cost per trade, 25 = 0.25%% (PancakeSwap fee tier)")
    ap.add_argument("--capital", type=float, default=1000)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--out", default=str(ROOT / "backtest.md"))
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    periods = [int(x) for x in args.periods.split(",") if x.strip()]
    md = run(cfg, periods, args.cost_bps, args.capital)
    Path(args.out).write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(cli())
