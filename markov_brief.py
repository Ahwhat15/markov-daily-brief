import os, time, requests, numpy as np, pandas as pd
from datetime import datetime, timedelta

TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
SEND_HOUR_UTC    = int(os.environ.get("SEND_HOUR_UTC", "8"))
CHECK_INTERVAL   = int(os.environ.get("CHECK_INTERVAL", "60"))

ASSETS = [
    {"name":"BTC-USD","emoji":"₿","ticker":"BTC-USD","window":20,"threshold":0.02,"vol_norm":False,"role":"Macro filter"},
    {"name":"ES/MES Futures","emoji":"📈","ticker":"ES=F","window":20,"threshold":0.04,"vol_norm":False,"role":"Market direction"},
]

def fetch_yfinance(ticker, years=10):
    try:
        import yfinance as yf
        end = datetime.utcnow()
        start = end - timedelta(days=years*365)
        df = yf.download(ticker, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), progress=False, auto_adjust=True)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df["Close"].dropna()
    except Exception as e:
        print(f"  yfinance error {ticker}: {e}")
        return None

def label_regimes(close, window, threshold, vol_norm=False):
    if vol_norm:
        roll_ret = close.pct_change(window)
        roll_std = close.pct_change().rolling(window).std()
        roll_std[roll_std==0] = 1e-9
        score = roll_ret / roll_std
        lbl = pd.Series(1, index=close.index, dtype=int)
        lbl[score > threshold] = 2
        lbl[score < -threshold] = 0
    else:
        roll = close.pct_change(window)
        lbl = pd.Series(1, index=close.index, dtype=int)
        lbl[roll > threshold] = 2
        lbl[roll < -threshold] = 0
    return lbl.dropna()

def build_matrix(labels):
    counts = np.zeros((3,3), dtype=float)
    arr = labels.to_numpy()
    for i in range(len(arr)-1):
        counts[arr[i], arr[i+1]] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums==0] = 1.0
    return counts / row_sums

def get_regime_data(asset):
    close = fetch_yfinance(asset["ticker"])
    if close is None or len(close) < 300: return None
    labels = label_regimes(close, asset["window"], asset["threshold"], asset["vol_norm"])
    P = build_matrix(labels)
    current = int(labels.iloc[-1])
    signal = float(P[current,2] - P[current,0])
    Pn = np.linalg.matrix_power(P, 5)
    forecast = Pn[current]
    state_names = {0:"Bear 🔴", 1:"Sideways ⚪", 2:"Bull 🟢"}
    return {
        "name":asset["name"],"emoji":asset["emoji"],"role":asset["role"],
        "regime":state_names[current],"signal":signal,
        "p_bull":P[current,2],"p_bear":P[current,0],"p_side":P[current,1],
        "f_bull":forecast[2],"f_bear":forecast[0],"f_side":forecast[1],
        "bull_stick":P[2,2],"bear_stick":P[0,0],
        "as_of":close.index[-1].strftime("%Y-%m-%d"),
    }

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id":TELEGRAM_CHAT_ID,"text":message,"parse_mode":"HTML"}
    try:
        r = requests.post(url, data=data, timeout=15)
        r.raise_for_status()
        print(f"  Telegram sent OK")
        return True
    except Exception as e:
        print(f"  Telegram error: {e}")
        return False

def signal_bar(signal):
    filled = max(0, min(10, round(abs(signal)*10)))
    bar = "█"*filled + "░"*(10-filled)
    direction = "LONG ▲" if signal > 0 else ("SHORT ▼" if signal < 0 else "FLAT —")
    return f"{bar}  {direction}"

def build_message(results):
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"<b>📊 MARKOV DAILY REGIME BRIEF</b>", f"<i>{now}</i>", ""]
    for r in results:
        if r is None: continue
        sig_arrow = "🔺" if r["signal"] > 0.2 else ("🔻" if r["signal"] < -0.2 else "➡️")
        lines += [
            f"<b>{r['emoji']} {r['name']}</b>  <i>({r['role']})</i>",
            f"  Regime today : {r['regime']}",
            f"  Signal       : {r['signal']:+.3f} {sig_arrow}",
            f"  {signal_bar(r['signal'])}",
            "",
            f"  Tomorrow's odds:",
            f"    🟢 Bull     {r['p_bull']*100:.1f}%",
            f"    🔴 Bear     {r['p_bear']*100:.1f}%",
            f"    ⚪ Sideways {r['p_side']*100:.1f}%",
            "",
            f"  5-day forecast:",
            f"    🟢 Bull     {r['f_bull']*100:.1f}%",
            f"    🔴 Bear     {r['f_bear']*100:.1f}%",
            f"    ⚪ Sideways {r['f_side']*100:.1f}%",
            "",
            f"  Stickiness   : Bull→Bull {r['bull_stick']*100:.0f}%  Bear→Bear {r['bear_stick']*100:.0f}%",
            f"  As of        : {r['as_of']}",
            "─"*32, "",
        ]
    signals = [r["signal"] for r in results if r is not None]
    if signals:
        avg = sum(signals)/len(signals)
        if avg > 0.15: bias = "🟢 RISK ON — Both signals positive"
        elif avg < -0.15: bias = "🔴 RISK OFF — Both signals negative"
        else: bias = "⚪ MIXED — Signals diverging, reduce size"
        lines += [f"<b>Overall bias: {bias}</b>", ""]
    lines.append("<i>VMc1 Investments — Markov Regime Monitor</i>")
    return "\n".join(lines)

def already_sent_today(last_sent):
    if last_sent is None: return False
    now = datetime.utcnow()
    return last_sent.date() == now.date() and now.hour >= SEND_HOUR_UTC

def main():
    print("="*60)
    print("  MARKOV DAILY BRIEF — VMc1 Investments")
    print(f"  Send time: {SEND_HOUR_UTC:02d}:00 UTC daily")
    print(f"  Assets: {', '.join(a['name'] for a in ASSETS)}")
    print("="*60)
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("ERROR: TELEGRAM_TOKEN and TELEGRAM_CHAT_ID must be set")
        return
    last_sent = None
    while True:
        now = datetime.utcnow()
        if now.hour == SEND_HOUR_UTC and not already_sent_today(last_sent):
            print(f"\n[{now.strftime('%Y-%m-%d %H:%M UTC')}] Sending daily brief...")
            results = []
            for asset in ASSETS:
                print(f"  Fetching {asset['name']}...")
                r = get_regime_data(asset)
                results.append(r)
                if r: print(f"    Regime: {r['regime']}  Signal: {r['signal']:+.3f}")
                else: print(f"    Failed to fetch data")
            message = build_message(results)
            success = send_telegram(message)
            if success: last_sent = now
        else:
            next_send = now.replace(hour=SEND_HOUR_UTC, minute=0, second=0, microsecond=0)
            if now.hour >= SEND_HOUR_UTC: next_send += timedelta(days=1)
            wait_mins = int((next_send-now).total_seconds()/60)
            print(f"[{now.strftime('%H:%M UTC')}] Waiting — next brief in {wait_mins} min")
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
