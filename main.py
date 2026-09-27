"""
Star Pattern Signal Bot — Morning Star (1H) / Evening Star (30min, 15min)
============================================================================
Posts Morning Star and Evening Star signals to a private Telegram channel,
tracks each as its own position, and posts TP/SL updates as they happen.
Does NOT auto-execute trades — signal posting only.

Strategy per timeframe (validated via backtesting):
  1h    -> Morning Star only | SL $15 | TP1 $15 / TP2 $30 / TP3 $45
  30min -> Evening Star only | SL $10 | TP1 $10 / TP2 $20 / TP3 $30
  15min -> Evening Star only | SL $10 | TP1 $15 (single target)

Position lifecycle:
  - 3-target strategies (1h, 30min): TP1 = notification only. Official
    result (win/loss) is TP2 (full close) vs SL.
  - 15min (single target): straightforward TP1-or-SL.

API budget (TwelveData free tier, 800 calls/day):
  - Signal scanning: 3 strategies x every 15 min = 288 calls/day (ALWAYS
    runs — this is how signals get found, can't be made conditional)
  - Price monitoring: only runs when at least one trade is open, so a
    day with zero signals stays near the 288 baseline; each concurrently
    open trade adds up to 480 calls/day at the 3-min interval

Deploy as a long-running process (Railway).
"""

import requests
import pandas as pd
import numpy as np
import json
import os
import time
from datetime import datetime, timezone

# ================== CONFIG ==================

TWELVEDATA_API_KEY = os.environ["TWELVEDATA_API_KEY"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHANNEL_ID = os.environ["TELEGRAM_CHANNEL_ID"]

SYMBOL = "XAU/USD"

STATE_FILE = "open_trades.json"   # persistence warning from earlier still applies

# --- Pattern shape thresholds (tuned/validated values) ---
STRONG_BODY_RATIO = 0.4
DOJI_MAX_BODY_RATIO = 0.20
GAP_TOLERANCE_DOLLARS = 1.5

# --- RSI / swing-location filter ---
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
SWING_ORDER = 3
SWING_LOOKBACK = 20
SWING_TOLERANCE_DOLLARS = 3.0

# --- How often the bot checks things (seconds) ---
SIGNAL_SCAN_INTERVAL = 900    # 15 min — always runs, this is the API floor
PRICE_MONITOR_INTERVAL = 180  # 3 min — only fires API calls when a trade is open
CANDLE_HISTORY_SIZE = 100

# --- The 3 strategies this bot runs, independently ---
STRATEGIES = {
    "1h_morning": {
        "interval": "1h", "pattern": "morning",
        "sl": 12, "tp_levels": [15, 30, 45],
        "scan_interval": 3600,   # 1 hour — matches this strategy's own candle close
    },
    "30min_evening": {
        "interval": "30min", "pattern": "evening",
        "sl": 10, "tp_levels": [10, 20, 30],
        "scan_interval": 1800,   # 30 min
    },
    "15min_evening": {
        "interval": "15min", "pattern": "evening",
        "sl": 10, "tp_levels": [15],
        "scan_interval": 900,    # 15 min
    },
}

# ================== TELEGRAM ==================

def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHANNEL_ID, "text": text, "parse_mode": "HTML"}
    try:
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code != 200:
            print(f"Telegram send failed: {r.status_code} {r.text}")
    except Exception as e:
        print(f"Telegram send error: {e}")


# ================== PERSISTENCE ==================

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {name: {"open_trade": None, "last_signal_time": None} for name in STRATEGIES}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def any_trade_open(state):
    return any(state[name]["open_trade"] is not None for name in STRATEGIES)


# ================== DATA FETCHING ==================

def fetch_recent_candles(interval, size):
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": SYMBOL, "interval": interval,
        "outputsize": size, "apikey": TWELVEDATA_API_KEY, "format": "JSON"
    }
    r = requests.get(url, params=params, timeout=30)
    data = r.json()
    if "values" not in data:
        print(f"Candle fetch error ({interval}): {data}")
        return None

    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    return df


def fetch_live_price():
    url = "https://api.twelvedata.com/price"
    params = {"symbol": SYMBOL, "apikey": TWELVEDATA_API_KEY}
    try:
        r = requests.get(url, params=params, timeout=15)
        data = r.json()
        if "price" in data:
            return float(data["price"])
    except Exception as e:
        print(f"Live price fetch error: {e}")
    return None


# ================== INDICATORS ==================

def add_indicators(df):
    df = df.copy()
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / RSI_PERIOD, min_periods=RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / RSI_PERIOD, min_periods=RSI_PERIOD, adjust=False).mean()
    rs = avg_gain / avg_loss
    df["rsi_14"] = 100 - (100 / (1 + rs))
    return df


def find_swings(df, order=SWING_ORDER):
    swing_highs, swing_lows = [], []
    closes = df["close"].values
    for i in range(order, len(df) - order):
        window = closes[i - order: i + order + 1]
        if closes[i] == window.max():
            swing_highs.append((i, i + order, closes[i]))
        if closes[i] == window.min():
            swing_lows.append((i, i + order, closes[i]))
    return swing_highs, swing_lows


def near_recent_swing(swings, as_of_index, price, lookback, tolerance):
    for swing_index, confirmed_at, swing_price in reversed(swings):
        if confirmed_at > as_of_index:
            continue
        if as_of_index - swing_index > lookback:
            break
        if abs(price - swing_price) <= tolerance:
            return True
    return False


# ================== PATTERN CHECK (with diagnostic reason) ==================

def check_pattern(df, kind):
    if len(df) < 30:
        return False, "not enough candle history yet (warming up)"

    df = add_indicators(df)
    swing_highs, swing_lows = find_swings(df)

    i3 = len(df) - 1
    i2 = i3 - 1
    i1 = i3 - 2
    candle_time = df["datetime"][i3]

    o1, c1, h1, l1 = df["open"][i1], df["close"][i1], df["high"][i1], df["low"][i1]
    o2, c2, h2, l2 = df["open"][i2], df["close"][i2], df["high"][i2], df["low"][i2]
    o3, c3 = df["open"][i3], df["close"][i3]

    range1, range2 = h1 - l1, h2 - l2
    if range1 <= 0 or range2 <= 0:
        return False, f"zero-range candle at {candle_time} (bad data?)"

    body1, body2 = abs(c1 - o1), abs(c2 - o2)
    midpoint1 = (o1 + c1) / 2
    rsi_now = df["rsi_14"][i3]

    if kind == "morning":
        if not (c1 < o1 and body1 >= STRONG_BODY_RATIO * range1):
            return False, f"Candle1 not a strong bearish candle (body {body1:.2f} / range {range1:.2f})"
        if not (body2 <= DOJI_MAX_BODY_RATIO * range2):
            return False, f"Candle2 not a doji (body {body2:.2f} / range {range2:.2f})"
        if not (max(o2, c2) <= c1 + GAP_TOLERANCE_DOLLARS):
            return False, "Candle2 not positioned at bottom of Candle1's body"
        if not (c3 > o3 and c3 > midpoint1):
            return False, f"Candle3 didn't close bullish past Candle1 midpoint ({midpoint1:.2f})"
        rsi_ok = rsi_now < RSI_OVERSOLD
        swing_ok = near_recent_swing(swing_lows, i2, l2, SWING_LOOKBACK, SWING_TOLERANCE_DOLLARS)
        if not (rsi_ok or swing_ok):
            return False, f"shape matched but RSI {rsi_now:.1f} not oversold and no nearby swing low"
        return True, f"Morning Star confirmed (RSI {rsi_now:.1f}, swing_low_nearby={swing_ok})"

    elif kind == "evening":
        if not (c1 > o1 and body1 >= STRONG_BODY_RATIO * range1):
            return False, f"Candle1 not a strong bullish candle (body {body1:.2f} / range {range1:.2f})"
        if not (body2 <= DOJI_MAX_BODY_RATIO * range2):
            return False, f"Candle2 not a doji (body {body2:.2f} / range {range2:.2f})"
        if not (min(o2, c2) >= c1 - GAP_TOLERANCE_DOLLARS):
            return False, "Candle2 not positioned at top of Candle1's body"
        if not (c3 < o3 and c3 < midpoint1):
            return False, f"Candle3 didn't close bearish past Candle1 midpoint ({midpoint1:.2f})"
        rsi_ok = rsi_now > RSI_OVERBOUGHT
        swing_ok = near_recent_swing(swing_highs, i2, h2, SWING_LOOKBACK, SWING_TOLERANCE_DOLLARS)
        if not (rsi_ok or swing_ok):
            return False, f"shape matched but RSI {rsi_now:.1f} not overbought and no nearby swing high"
        return True, f"Evening Star confirmed (RSI {rsi_now:.1f}, swing_high_nearby={swing_ok})"

    return False, "unknown pattern kind"


# ================== SIGNAL SCANNING (with logging) ==================

def scan_for_signal(strategy_name, cfg, state):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    try:
        slot = state[strategy_name]
        if slot["open_trade"] is not None:
            print(f"[{ts}] [{strategy_name}] skip scan — trade already open")
            return

        df = fetch_recent_candles(cfg["interval"], CANDLE_HISTORY_SIZE)
        if df is None or len(df) < 30:
            print(f"[{ts}] [{strategy_name}] ERROR — candle fetch failed or insufficient data")
            return

        latest_candle_time = str(df["datetime"].iloc[-1])
        if slot["last_signal_time"] == latest_candle_time:
            print(f"[{ts}] [{strategy_name}] no new closed candle since last scan ({latest_candle_time})")
            return

        slot["last_signal_time"] = latest_candle_time

        found, reason = check_pattern(df, cfg["pattern"])
        print(f"[{ts}] [{strategy_name}] scanned candle {latest_candle_time} — {reason}")

        if found:
            entry = float(df["close"].iloc[-1])
            direction = "long" if cfg["pattern"] == "morning" else "short"
            tp_levels = cfg["tp_levels"]
            sl_dollars = cfg["sl"]

            if direction == "long":
                sl_price = entry - sl_dollars
                tp_prices = [entry + t for t in tp_levels]
            else:
                sl_price = entry + sl_dollars
                tp_prices = [entry - t for t in tp_levels]

            trade = {
                "strategy": strategy_name,
                "direction": direction,
                "entry": entry,
                "sl": sl_price,
                "tp_prices": tp_prices,
                "tp_dollars": tp_levels,
                "tp1_hit": False,
                "tp2_hit": False,
                "opened_at": datetime.now(timezone.utc).isoformat(),
            }
            slot["open_trade"] = trade
            save_state(state)

            pattern_label = "Morning Star (Bullish)" if cfg["pattern"] == "morning" else "Evening Star (Bearish)"
            tp_lines = "\n".join([f"TP{i+1}: ${p:.2f}" for i, p in enumerate(tp_prices)])
            msg = (
                f"🌟 <b>NEW SIGNAL — {strategy_name}</b>\n"
                f"{pattern_label}\n\n"
                f"Direction: {'BUY' if direction == 'long' else 'SELL'}\n"
                f"Entry: ${entry:.2f}\n"
                f"SL: ${sl_price:.2f}\n"
                f"{tp_lines}"
            )
            send_telegram_message(msg)
            print(f"[{ts}] [{strategy_name}] *** SIGNAL SENT *** {direction} @ {entry}")
        else:
            save_state(state)

    except Exception as e:
        print(f"[{ts}] [{strategy_name}] EXCEPTION during scan: {e}")


# ================== TRADE MONITORING ==================

def monitor_open_trade(strategy_name, cfg, state, live_price):
    slot = state[strategy_name]
    trade = slot["open_trade"]
    if trade is None or live_price is None:
        return

    direction = trade["direction"]
    sl = trade["sl"]
    tp_prices = trade["tp_prices"]
    n_tps = len(tp_prices)

    def level_hit(level):
        return live_price >= level if direction == "long" else live_price <= level

    def sl_hit():
        return live_price <= sl if direction == "long" else live_price >= sl

    if sl_hit():
        send_telegram_message(f"🔴 <b>{strategy_name}</b> — SL hit @ ${live_price:.2f}. Result: LOSS")
        slot["open_trade"] = None
        save_state(state)
        return

    if n_tps == 1:
        if level_hit(tp_prices[0]):
            send_telegram_message(f"🟢 <b>{strategy_name}</b> — TP1 hit @ ${live_price:.2f}. Result: WIN")
            slot["open_trade"] = None
            save_state(state)
        return

    if not trade["tp1_hit"] and level_hit(tp_prices[0]):
        trade["tp1_hit"] = True
        send_telegram_message(f"🟡 <b>{strategy_name}</b> — TP1 hit @ ${live_price:.2f} (still running to TP2)")
        save_state(state)

    if not trade["tp2_hit"] and level_hit(tp_prices[1]):
        trade["tp2_hit"] = True
        send_telegram_message(f"🟢 <b>{strategy_name}</b> — TP2 hit @ ${live_price:.2f}. Result: WIN (official close)")
        slot["open_trade"] = None
        save_state(state)
        return


# ================== MAIN LOOP ==================

def main():
    state = load_state()
    send_telegram_message("✅ Bot started — monitoring....")

    last_scan_time = {name: 0 for name in STRATEGIES}
    last_monitor_time = 0

    while True:
        try:
            now = time.time()
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

            # --- Price monitoring: only spends an API call if a trade is actually open ---
            if now - last_monitor_time >= PRICE_MONITOR_INTERVAL:
                if any_trade_open(state):
                    live_price = fetch_live_price()
                    for name, cfg in STRATEGIES.items():
                        monitor_open_trade(name, cfg, state, live_price)
                else:
                    print(f"[{ts}] price monitor skipped — no open trades")
                last_monitor_time = now

            # --- Signal scanning: each strategy on its OWN cadence now ---
            for name, cfg in STRATEGIES.items():
                if now - last_scan_time[name] >= cfg["scan_interval"]:
                    scan_for_signal(name, cfg, state)
                    last_scan_time[name] = now

        except Exception as e:
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{ts}] Main loop error: {e}")

        time.sleep(10)


if __name__ == "__main__":
    main()
