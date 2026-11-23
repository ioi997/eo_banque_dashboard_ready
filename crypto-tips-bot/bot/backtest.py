#!/usr/bin/env python3
"""
Backtest the daily tips bot over the last N days.

For every past day it recomputes the signals using ONLY the data available that day
(no look-ahead), then measures:
  1. Strategy "follow the bot": each day hold, in equal weights, every asset flagged as a
     buy (score >= 1 and RSI <= 75); otherwise sit in cash (stablecoins). Trading costs included.
  2. Benchmark "buy and hold": equal weight in the whole watchlist from day 1.
  3. Signal quality: what happened 1 and 5 days after each type of signal.

Usage: python bot/backtest.py --days 30
Not financial advice. 30 days is a very small sample.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import main  # noqa: E402

ROOT = main.ROOT


def is_buy(sig: main.Signal) -> bool:
    return sig.score >= 1 and sig.rsi <= 75


def bucket(sig: main.Signal) -> str:
    if is_buy(sig):
        return "🟢 Buy / accumulate"
    if sig.score >= 1:
        return "🟡 Overheated, wait"
    if sig.score == 0:
        return "⚪ Hold"
    if sig.score >= -2:
        return "🟠 Caution"
    return "🔴 Weak trend"


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min() * 100)


def run(cfg: dict, days: int, cost_bps: float, start_capital: float) -> str:
    min_days = int(cfg.get("settings", {}).get("min_history_days", 210))

    hist: dict[str, pd.DataFrame] = {}
    meta: dict[str, dict] = {}
    for a in cfg["assets"]:
        df = main.get_history(a)
        if df is None or len(df) < min_days + days:
            print(f"skip {a['name']} (not enough data)", file=sys.stderr)
            continue
        hist[a["name"]], meta[a["name"]] = df, a

    if not hist:
        raise SystemExit("No data downloaded.")

    # Common calendar: every day any asset traded; ETFs keep their last close on weekends.
    end = max(df.index[-1] for df in hist.values())
    start = end - pd.Timedelta(days=days)
    calendar = sorted({d for df in hist.values() for d in df.index if d > start})
    closes = pd.DataFrame({n: df["Close"] for n, df in hist.items()}).sort_index().ffill()
    closes = closes.reindex(closes.index.union(calendar)).ffill()

    # Signals for each decision day (the day BEFORE each calendar day), no look-ahead.
    decision_days = [d for d in closes.index if d < calendar[-1] and d >= closes.index[closes.index.get_loc(calendar[0]) - 1]]
    signals: dict[pd.Timestamp, dict[str, main.Signal]] = {}
    for d in decision_days:
        signals[d] = {}
        for n, df in hist.items():
            past = df[df.index <= d]
            if len(past) >= min_days:
                signals[d][n] = main.analyse(meta[n], past)

    names = list(hist)
    rets = closes[names].pct_change().fillna(0)

    # --- Strategy vs benchmark ------------------------------------------------
    eq_bot, eq_bh = [start_capital], [start_capital]
    prev_w = pd.Series(0.0, index=names)
    trades, invested_days = 0, 0
    log_rows = []
    for i in range(1, len(decision_days) + 1):
        d_prev = decision_days[i - 1]
        d = closes.index[closes.index.get_loc(d_prev) + 1]
        picks = [n for n, s in signals[d_prev].items() if is_buy(s)]
        w = pd.Series(0.0, index=names)
        if picks:
            w[picks] = 1 / len(picks)
            invested_days += 1
        turnover = float((w - prev_w).abs().sum())
        trades += int(((w > 0) != (prev_w > 0)).sum())
        r_bot = float((w * rets.loc[d]).sum()) - turnover * cost_bps / 10_000
        r_bh = float(rets.loc[d, names].mean())
        eq_bot.append(eq_bot[-1] * (1 + r_bot))
        eq_bh.append(eq_bh[-1] * (1 + r_bh))
        log_rows.append((d.date(), ", ".join(picks) or "cash", r_bot * 100, r_bh * 100))
        prev_w = w

    eq_bot_s, eq_bh_s = pd.Series(eq_bot), pd.Series(eq_bh)
    bot_ret = (eq_bot[-1] / start_capital - 1) * 100
    bh_ret = (eq_bh[-1] / start_capital - 1) * 100

    # --- Signal quality ------------------------------------------------------
    rows = []
    for d, sigs in signals.items():
        loc = closes.index.get_loc(d)
        for n, s in sigs.items():
            p0 = closes[n].iloc[loc]
            f1 = closes[n].iloc[loc + 1] / p0 - 1 if loc + 1 < len(closes) else np.nan
            f5 = closes[n].iloc[loc + 5] / p0 - 1 if loc + 5 < len(closes) else np.nan
            rows.append({"asset": n, "bucket": bucket(s), "buy": is_buy(s),
                         "f1": f1 * 100, "f5": f5 * 100})
    q = pd.DataFrame(rows)

    # --- Per asset -------------------------------------------------------------
    per_asset = []
    for n in names:
        a_ret = (closes[n].iloc[-1] / closes[n].loc[decision_days[0]] - 1) * 100
        days_buy = int(q[(q.asset == n)].buy.sum())
        last = signals[decision_days[-1]].get(n)
        per_asset.append((n, a_ret, days_buy, bucket(last) if last else "n/a"))

    # --- Report ------------------------------------------------------------------
    diff = bot_ret - bh_ret
    if diff > 1:
        verdict = f"✅ The bot **beat** buy and hold by {diff:.1f} points."
    elif diff < -1:
        verdict = f"❌ The bot **lagged** buy and hold by {abs(diff):.1f} points."
    else:
        verdict = "➖ The bot performed **about the same** as buy and hold."
    dd_bot, dd_bh = max_drawdown(eq_bot_s), max_drawdown(eq_bh_s)
    if dd_bot > dd_bh + 1:
        verdict += f" It also cut the worst drop from {dd_bh:.1f}% to {dd_bot:.1f}%."

    L = [f"# 🧪 Backtest: last {days} days (to {end.date()})", "", verdict, "",
         "| | Follow the bot | Buy and hold all |", "|---|---:|---:|",
         f"| Start capital | {start_capital:,.0f} | {start_capital:,.0f} |",
         f"| End value | {eq_bot[-1]:,.0f} | {eq_bh[-1]:,.0f} |",
         f"| Return | {main.fmt_pct(bot_ret)} | {main.fmt_pct(bh_ret)} |",
         f"| Worst drop (max drawdown) | {dd_bot:.1f}% | {dd_bh:.1f}% |",
         f"| Days invested | {invested_days}/{len(decision_days)} | {len(decision_days)}/{len(decision_days)} |",
         f"| Position changes | {trades} | 0 |",
         f"| Trading cost assumed | {cost_bps:.0f} bps per trade | none |", ""]

    L += ["## 🎯 Signal quality", "",
          "Average price move after each signal type (higher is better for buys, lower is better for warnings).", "",
          "| Signal | Count | Avg next day | Avg next 5 days | % up after 5 days |",
          "|---|---:|---:|---:|---:|"]
    for b, g in q.groupby("bucket"):
        g5 = g.f5.dropna()
        L.append(f"| {b} | {len(g)} | {main.fmt_pct(g.f1.mean())} | "
                 f"{main.fmt_pct(g5.mean()) if len(g5) else 'n/a'} | "
                 f"{(g5 > 0).mean() * 100:.0f}% |" if len(g5) else
                 f"| {b} | {len(g)} | {main.fmt_pct(g.f1.mean())} | n/a | n/a |")
    L.append("")

    L += ["## 📦 Per asset", "", "| Asset | Price change | Days flagged buy | Signal today |",
          "|---|---:|---:|---|"]
    for n, r, db, b in sorted(per_asset, key=lambda x: -x[1]):
        L.append(f"| {n} | {main.fmt_pct(r)} | {db} | {b} |")
    L.append("")

    L += ["<details><summary>Daily log</summary>", "",
          "| Day | Held | Bot | Buy and hold |", "|---|---|---:|---:|"]
    L += [f"| {d} | {p} | {main.fmt_pct(rb)} | {main.fmt_pct(rh)} |" for d, p, rb, rh in log_rows]
    L += ["", "</details>", "", "---",
          "_Signals recomputed each day with only the data known that day. Positions are taken at the "
          "close after the signal and held one day. ETFs are flat on weekends. 30 days is a tiny sample: "
          "judge the bot on several months before trusting it. Not financial advice._"]
    return "\n".join(L)


def cli() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--cost-bps", type=float, default=25, help="cost per trade, 25 = 0.25%% (PancakeSwap fee tier)")
    ap.add_argument("--capital", type=float, default=1000)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--out", default=str(ROOT / "backtest.md"))
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    md = run(cfg, args.days, args.cost_bps, args.capital)
    Path(args.out).write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(cli())
