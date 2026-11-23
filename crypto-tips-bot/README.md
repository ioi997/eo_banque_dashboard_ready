# 📈 Daily Crypto & ETF Tips Bot

A free bot that runs on GitHub Actions every morning, scans your watchlist
(crypto + ETFs behind the tokenized versions you trade on BSC) and sends you
a daily report with:

- **Best setups**: assets in an uptrend that are pulling back (good DCA moments)
- **Be careful with**: assets in a downtrend or overheated
- **Alerts**: golden / death crosses, RSI bounces, 52-week highs
- **Full table**: price, 1d / 1w / 1m change, RSI, distance from 52w high, score
- **Levels**: 20-day support / resistance and a suggested stop / review level (2x ATR)
- **Crypto Fear & Greed index**

Delivered as a **GitHub issue** (GitHub emails you automatically) and optionally on **Telegram**.
Every report is also saved in `reports/YYYY-MM-DD.md` so you keep a history.

> Automated technical signals for information only. Not financial advice.

---

## Setup (about 5 minutes)

1. **Create a repo** on GitHub (private is fine), e.g. `crypto-tips-bot`.
2. **Upload these files** (drag and drop the unzipped folder in "Add file > Upload files").
   Make sure `.github/workflows/daily-tips.yml` is included (hidden folder on Mac: press Cmd+Shift+. to show it).
3. Go to **Settings > Actions > General > Workflow permissions** and select **Read and write permissions**. Save.
4. Go to the **Actions** tab, enable workflows if asked, open **Daily market tips** and click **Run workflow** to test it now.
5. Check the **Issues** tab: your first report is there. To get it by email, click **Watch > All activity** on the repo.

From then on it runs every day at 07:30 Paris time (08:30 in summer).

### Optional: Telegram notifications

1. In Telegram, message **@BotFather**, send `/newbot`, follow the steps and copy the **token**.
2. Send any message to your new bot, then open
   `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates` in a browser and copy the `"chat":{"id": ...}` number.
3. In the repo: **Settings > Secrets and variables > Actions > New repository secret**:
   - `TELEGRAM_BOT_TOKEN` = your token
   - `TELEGRAM_CHAT_ID` = your chat id

---

## Backtest: would it have made money?

Go to **Actions > Backtest > Run workflow**, choose the number of days (default 30) and a starting
capital. A few minutes later an issue shows:

- **Follow the bot** (hold every asset flagged as a buy, cash otherwise, 0.25% fee per trade)
  vs **buy and hold** the whole watchlist: return, worst drop, number of trades
- **Signal quality**: how prices moved 1 and 5 days after each signal type
- **Per asset** results and a daily log

Signals are recomputed each day with only the data known that day (no cheating with hindsight).
Try 30, 90 and 180 days: 30 days alone is too short to judge.

## Customise

- **Watchlist**: edit `config.yaml`. Add any Yahoo Finance symbol (`AAPL`, `TSLA`, `XRP-USD`, `URTH`...).
  Use `trade_as` to note how you actually trade it (Ondo token, bStocks, Binance...).
- **Time**: change the `cron` line in `.github/workflows/daily-tips.yml` (UTC time).
- **How many picks**: `top_n` in `config.yaml`.

## How the score works

| Rule | Points |
|---|---|
| Price above / below 200-day average | +1 / -1 |
| 50-day average above / below 200-day | +1 / -1 |
| MACD momentum improving / fading | +1 / -1 |
| RSI < 30 (oversold) | +2 in uptrend, +1 otherwise |
| RSI 30 to 45 in an uptrend (healthy pullback) | +1 |
| RSI > 70 / > 80 (overheated) | -1 / -2 |

| Score | Signal |
|---|---|
| 3 or more | 🟢 Strong setup, accumulate |
| 1 to 2 | 🟢 Accumulate / DCA |
| 0 | ⚪ Hold, no action |
| -1 to -2 | 🟠 Caution, pause new buys |
| -3 or less | 🔴 Weak trend, protect gains |

Uptrends with RSI above 75 show 🟡 "wait for a pullback" instead of a buy signal.

The logic suits a **buy and hold / DCA** style: buy quality assets on dips while the
long-term trend is up, avoid chasing spikes, and slow down when trends break.

## Notes

- ETF signals use the last US market close. Tokenized versions (Ondo, bStocks, xStocks) track
  the underlying but can trade at a premium or discount in DEX pools: check before buying.
- Data: Yahoo Finance, with CoinGecko (crypto) and Stooq (ETFs) as fallbacks. No API key needed.
- GitHub pauses scheduled workflows after 60 days without repo activity; the daily report commit keeps it active.
- Run locally: `pip install -r requirements.txt && python bot/main.py`
