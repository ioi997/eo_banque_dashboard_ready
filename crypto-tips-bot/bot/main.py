#!/usr/bin/env python3
"""
Daily tips bot for crypto and ETFs.

Fetches daily prices, computes trend / momentum / RSI signals, scores each asset
and writes a Markdown report (plus an optional short Telegram message).

These are rule-based technical signals, NOT financial advice.
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parent.parent
HEADERS = {"User-Agent": "Mozilla/5.0 (daily-tips-bot)"}


# --------------------------------------------------------------------------
# Data sources (Yahoo first, then free fallbacks)
# --------------------------------------------------------------------------
def _clean(df: pd.DataFrame) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[cols].dropna(subset=["Close"])
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.sort_index()


def fetch_yahoo(symbol: str) -> pd.DataFrame | None:
    try:
        import yfinance as yf

        for attempt in range(3):
            df = yf.download(symbol, period="2y", interval="1d", progress=False,
                             auto_adjust=True, threads=False)
            df = _clean(df)
            if df is not None and len(df) > 50:
                return df
            time.sleep(2 * (attempt + 1))
    except Exception as e:  # noqa: BLE001
        print(f"[yahoo] {symbol}: {e}", file=sys.stderr)
    return None


def fetch_coingecko(cg_id: str) -> pd.DataFrame | None:
    url = (f"https://api.coingecko.com/api/v3/coins/{cg_id}/market_chart"
           f"?vs_currency=usd&days=365&interval=daily")
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        prices = r.json()["prices"]
        df = pd.DataFrame(prices, columns=["ts", "Close"])
        df.index = pd.to_datetime(df["ts"], unit="ms")
        df = df[["Close"]]
        df["Open"] = df["High"] = df["Low"] = df["Close"]  # close-only fallback
        return _clean(df)
    except Exception as e:  # noqa: BLE001
        print(f"[coingecko] {cg_id}: {e}", file=sys.stderr)
        return None


def fetch_stooq(symbol: str) -> pd.DataFrame | None:
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), parse_dates=["Date"], index_col="Date")
        return _clean(df)
    except Exception as e:  # noqa: BLE001
        print(f"[stooq] {symbol}: {e}", file=sys.stderr)
        return None


def get_history(asset: dict) -> pd.DataFrame | None:
    df = fetch_yahoo(asset["yahoo"]) if asset.get("yahoo") else None
    if df is None and asset.get("coingecko"):
        df = fetch_coingecko(asset["coingecko"])
    if df is None and asset.get("stooq"):
        df = fetch_stooq(asset["stooq"])
    return df


def fetch_fear_greed() -> tuple[int, str] | None:
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", headers=HEADERS, timeout=20)
        d = r.json()["data"][0]
        return int(d["value"]), d["value_classification"]
    except Exception as e:  # noqa: BLE001
        print(f"[fear&greed] {e}", file=sys.stderr)
        return None


# --------------------------------------------------------------------------
# Indicators
# --------------------------------------------------------------------------
def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100)


def macd_hist(close: pd.Series) -> pd.Series:
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    return macd - macd.ewm(span=9, adjust=False).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df["Close"].shift()
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - prev).abs(),
                    (df["Low"] - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


# --------------------------------------------------------------------------
# Signal engine
# --------------------------------------------------------------------------
@dataclass
class Signal:
    name: str
    type: str
    trade_as: str
    price: float
    chg_1d: float
    chg_7d: float
    chg_30d: float
    rsi: float
    from_high: float
    score: int
    verdict: str
    stop: float
    support: float
    resistance: float
    reasons: list[str] = field(default_factory=list)
    alerts: list[str] = field(default_factory=list)


VERDICTS = [
    (3, "🟢 Strong setup, accumulate"),
    (1, "🟢 Accumulate / DCA"),
    (0, "⚪ Hold, no action"),
    (-2, "🟠 Caution, pause new buys"),
    (-99, "🔴 Weak trend, protect gains"),
]


def verdict_for(score: int) -> str:
    for threshold, label in VERDICTS:
        if score >= threshold:
            return label
    return VERDICTS[-1][1]


def pct(a: float, b: float) -> float:
    return (a / b - 1) * 100 if b else 0.0


def analyse(asset: dict, df: pd.DataFrame) -> Signal:
    c = df["Close"]
    sma50, sma200 = c.rolling(50).mean(), c.rolling(200).mean()
    r = rsi(c)
    h = macd_hist(c)
    a = atr(df)

    price = float(c.iloc[-1])
    last_rsi = float(r.iloc[-1])
    up_long = price > sma200.iloc[-1]
    golden = sma50.iloc[-1] > sma200.iloc[-1]

    score, reasons, alerts = 0, [], []

    # 1) Long-term trend
    if up_long:
        score += 1; reasons.append("above 200-day average (long-term uptrend)")
    else:
        score -= 1; reasons.append("below 200-day average (long-term downtrend)")

    # 2) Medium-term trend
    if golden:
        score += 1; reasons.append("50d above 200d")
    else:
        score -= 1; reasons.append("50d below 200d")

    # 3) Momentum (MACD histogram direction)
    if h.iloc[-1] > 0 and h.iloc[-1] > h.iloc[-2]:
        score += 1; reasons.append("momentum improving")
    elif h.iloc[-1] < 0 and h.iloc[-1] < h.iloc[-2]:
        score -= 1; reasons.append("momentum fading")

    # 4) RSI: buy dips in uptrends, avoid chasing overheated moves
    if last_rsi < 30:
        score += 2 if up_long else 1
        reasons.append(f"oversold (RSI {last_rsi:.0f})")
    elif last_rsi < 45 and up_long:
        score += 1; reasons.append(f"pullback in uptrend (RSI {last_rsi:.0f})")
    elif last_rsi > 80:
        score -= 2; reasons.append(f"very overbought (RSI {last_rsi:.0f}), don't chase")
    elif last_rsi > 70:
        score -= 1; reasons.append(f"overbought (RSI {last_rsi:.0f})")

    # Event alerts over the last 5 sessions
    cross = (sma50 > sma200).astype(int).diff().iloc[-5:]
    if (cross == 1).any():
        alerts.append("Golden cross (50d crossed above 200d) in the last 5 days")
    if (cross == -1).any():
        alerts.append("Death cross (50d crossed below 200d) in the last 5 days")
    if r.iloc[-2] < 30 <= last_rsi:
        alerts.append("RSI bounced back above 30 (possible bottom)")
    if r.iloc[-2] > 70 >= last_rsi:
        alerts.append("RSI dropped back below 70 (momentum cooling)")
    hi52 = float(c.iloc[-252:].max())
    if price >= hi52 * 0.995:
        alerts.append("Trading at a 52-week high")

    def back(n: int) -> float:
        return float(c.iloc[-1 - n]) if len(c) > n else float(c.iloc[0])

    return Signal(
        name=asset["name"], type=asset["type"], trade_as=asset.get("trade_as", ""),
        price=price,
        chg_1d=pct(price, back(1)), chg_7d=pct(price, back(7 if asset["type"] == "crypto" else 5)),
        chg_30d=pct(price, back(30 if asset["type"] == "crypto" else 21)),
        rsi=last_rsi, from_high=pct(price, hi52),
        score=score,
        verdict=("🟡 Uptrend but overheated, wait for a pullback"
                 if last_rsi > 75 and score >= 1 else verdict_for(score)),
        stop=price - 2 * float(a.iloc[-1]),
        support=float(df["Low"].iloc[-20:].min()),
        resistance=float(df["High"].iloc[-20:].max()),
        reasons=reasons, alerts=alerts,
    )


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------
def fmt_price(p: float) -> str:
    if p >= 1000:
        return f"{p:,.0f}"
    if p >= 1:
        return f"{p:,.2f}"
    return f"{p:.4f}"


def fmt_pct(x: float) -> str:
    return f"{'+' if x >= 0 else ''}{x:.1f}%"


def build_markdown(signals: list[Signal], failed: list[str], fng, top_n: int, today: str) -> str:
    ranked = sorted(signals, key=lambda s: s.score, reverse=True)
    L = [f"# 📈 Daily market tips, {today}", ""]

    if fng:
        v, label = fng
        hint = ("extreme fear is historically a better time to accumulate slowly" if v <= 25 else
                "extreme greed: be careful adding size here" if v >= 75 else
                "neutral zone")
        L += [f"**Crypto Fear & Greed:** {v}/100 ({label}), {hint}.", ""]

    best = [s for s in ranked if s.score >= 1 and s.rsi <= 75][:top_n]
    L.append("## ✅ Best setups today")
    if best:
        for s in best:
            L.append(f"- **{s.name}** ({fmt_price(s.price)}): {s.verdict}. "
                     f"{'; '.join(s.reasons)}. Trade as: {s.trade_as}. "
                     f"Suggested stop / review level: {fmt_price(s.stop)}")
    else:
        L.append("- Nothing scores positive today. Patience is a position: keep cash / stablecoins.")
    L.append("")

    weak = [s for s in ranked[::-1] if s.score <= -2][:top_n]
    if weak:
        L.append("## ⚠️ Be careful with")
        for s in weak:
            L.append(f"- **{s.name}** ({fmt_price(s.price)}): {s.verdict}. {'; '.join(s.reasons)}.")
        L.append("")

    alerts = [(s.name, a) for s in ranked for a in s.alerts]
    if alerts:
        L.append("## 🔔 Alerts")
        L += [f"- **{n}**: {a}" for n, a in alerts]
        L.append("")

    L += ["## 📊 Full watchlist", "",
          "| Asset | Price | 1d | 1w | 1m | RSI | From 52w high | Score | Signal |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for s in ranked:
        L.append(f"| {s.name} | {fmt_price(s.price)} | {fmt_pct(s.chg_1d)} | {fmt_pct(s.chg_7d)} | "
                 f"{fmt_pct(s.chg_30d)} | {s.rsi:.0f} | {fmt_pct(s.from_high)} | {s.score:+d} | {s.verdict} |")
    L.append("")

    L += ["<details><summary>Levels (20-day support / resistance, 2xATR stop)</summary>", "",
          "| Asset | Support | Resistance | Stop / review |", "|---|---:|---:|---:|"]
    for s in ranked:
        L.append(f"| {s.name} | {fmt_price(s.support)} | {fmt_price(s.resistance)} | {fmt_price(s.stop)} |")
    L += ["", "</details>", ""]

    if failed:
        L += [f"_No data today for: {', '.join(failed)}_", ""]

    L += ["---",
          "_How the score works: +1/-1 for price vs 200-day average, +1/-1 for 50d vs 200d, "
          "+1/-1 for momentum, RSI adds up to +2 on dips in uptrends and up to -2 when overheated. "
          "ETF data is from the last US close. Tokenized versions may trade at a premium or discount; "
          "check the pool price before buying._",
          "",
          "_Automated technical signals for information only. Not financial advice._"]
    return "\n".join(L)


def build_telegram(signals: list[Signal], fng, top_n: int, today: str) -> str:
    ranked = sorted(signals, key=lambda s: s.score, reverse=True)
    lines = [f"📈 <b>Daily tips {today}</b>"]
    if fng:
        lines.append(f"Fear &amp; Greed: {fng[0]} ({fng[1]})")
    lines.append("")
    best = [s for s in ranked if s.score >= 1 and s.rsi <= 75][:top_n]
    lines.append("<b>Best setups</b>")
    lines += [f"• {s.name} {fmt_price(s.price)} | {s.verdict} | stop {fmt_price(s.stop)}" for s in best] \
        or ["• None today, stay patient"]
    weak = [s for s in ranked[::-1] if s.score <= -2][:top_n]
    if weak:
        lines += ["", "<b>Careful</b>"] + [f"• {s.name}: {s.verdict}" for s in weak]
    alerts = [f"• {s.name}: {a}" for s in ranked for a in s.alerts]
    if alerts:
        lines += ["", "<b>Alerts</b>"] + alerts[:8]
    lines += ["", "<i>Not financial advice.</i>"]
    return "\n".join(lines)


def send_telegram(text: str) -> None:
    token, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("Telegram not configured, skipping.")
        return
    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      json={"chat_id": chat, "text": text[:4000], "parse_mode": "HTML",
                            "disable_web_page_preview": True}, timeout=30)
    print("Telegram:", r.status_code, r.text[:200])


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--out", default=str(ROOT / "report.md"))
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    settings = cfg.get("settings", {})
    min_days = int(settings.get("min_history_days", 210))
    top_n = int(settings.get("top_n", 3))
    today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")

    signals, failed = [], []
    for asset in cfg["assets"]:
        df = get_history(asset)
        if df is None or len(df) < min_days:
            failed.append(asset["name"])
            continue
        try:
            signals.append(analyse(asset, df))
            print(f"OK  {asset['name']}")
        except Exception as e:  # noqa: BLE001
            print(f"ERR {asset['name']}: {e}", file=sys.stderr)
            failed.append(asset["name"])

    if not signals:
        print("No data for any asset, aborting.", file=sys.stderr)
        return 1

    fng = fetch_fear_greed()
    md = build_markdown(signals, failed, fng, top_n, today)
    Path(args.out).write_text(md, encoding="utf-8")

    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / f"{today}.md").write_text(md, encoding="utf-8")

    send_telegram(build_telegram(signals, fng, top_n, today))
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
