"""
Requirements:
pip install nsepython pandas numpy yfinance requests
"""

import pandas as pd
import numpy as np
import time, traceback, os
from datetime import datetime, timedelta
from nsepython import nse_optionchain_scrapper
import requests

# yfinance fallback
try:
    import yfinance as yf
    HAVE_YFINANCE = True
except:
    HAVE_YFINANCE = False

# ---------- CONFIG ----------
MIN_OI = 5000
MIN_VOL = 1000
STRIKES_AROUND_ATM = 1
TARGET_MULTIPLIER = 1.6
SL_MULTIPLIER = 0.6
REQUEST_DELAY = 1

RSI_PERIOD = 14
RSI_THRESHOLD = 50
STRICT_TREND_GATING = True
DF_MIN_CONF = 0.8

BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

ACTIVE_FILE = "active_suggestions.csv"
HISTORY_FILE = "all_history.csv"
# ----------------------------

def telegram_send(message):
    if not BOT_TOKEN or not CHAT_ID:
        print("[!] Telegram BOT_TOKEN or CHAT_ID missing")
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        requests.get(url, params={"chat_id":CHAT_ID,"text":message})
    except Exception as e:
        print(f"[!] Telegram send failed: {e}")

def get_nifty50_list():
    try:
        url = "https://archives.nseindia.com/content/indices/ind_nifty50list.csv"
        df = pd.read_csv(url)
        return df['Symbol'].astype(str).str.strip().tolist()
    except:
        return ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]

def pick_atm_strikes(strikes, spot):
    nearest = min(strikes, key=lambda x: abs(x - spot))
    idx = strikes.index(nearest)
    left = max(0, idx - STRIKES_AROUND_ATM)
    right = min(len(strikes)-1, idx + STRIKES_AROUND_ATM)
    return strikes[left:right+1]

def compute_rsi(series, period=14):
    delta = series.diff().dropna()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_sma(series, window):
    return series.rolling(window=window).mean()

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal

def fetch_history(symbol, days=120):
    if HAVE_YFINANCE:
        try:
            ticker = f"{symbol}.NS"
            end = datetime.today()
            start = end - timedelta(days=days+20)
            kl = yf.download(ticker, start=start, end=end, progress=False)
            if not kl.empty:
                return kl[['Close']].copy()
        except:
            pass
    return None

def trend_confirmations(symbol):
    res = {
        'price_gt_yesterday': None,
        'rsi': None, 'rsi_ok': None,
        'sma20_gt_sma50': None,
        'macd': None, 'macd_signal': None, 'macd_ok': None,
        'bias': None
    }
    hist = fetch_history(symbol, days=120)
    if hist is None or hist.empty:
        return res
    closes = hist['Close'].dropna()
    if len(closes) < 60:
        return res

    yclose = closes.iloc[-2].item()
    lclose = closes.iloc[-1].item()
    res['price_gt_yesterday'] = (lclose > yclose)
    rsi_val = compute_rsi(closes, RSI_PERIOD).iloc[-1].item()
    res['rsi'] = round(rsi_val, 2)
    res['rsi_ok'] = rsi_val > RSI_THRESHOLD
    sma20 = compute_sma(closes, 20).iloc[-1].item()
    sma50 = compute_sma(closes, 50).iloc[-1].item()
    res['sma20_gt_sma50'] = sma20 > sma50
    macd, macd_signal = compute_macd(closes)
    res['macd'] = macd.iloc[-1].item()
    res['macd_signal'] = macd_signal.iloc[-1].item()
    res['macd_ok'] = res['macd'] > res['macd_signal']

    bullish = (res['price_gt_yesterday'] and res['rsi_ok'] and res['sma20_gt_sma50'] and res['macd_ok'])
    bearish = ((not res['price_gt_yesterday']) and (rsi_val < 100 - RSI_THRESHOLD)
               and (not res['sma20_gt_sma50']) and (not res['macd_ok']))

    if bullish:
        res['bias'] = "BULLISH"
    elif bearish:
        res['bias'] = "BEARISH"
    else:
        res['bias'] = "NEUTRAL"

    return res

def analyze_option_chain(symbol):
    try:
        data = nse_optionchain_scrapper(symbol)
        spot = float(data['records']['underlyingValue'])
    except:
        return []

    df = []
    for row in data['records']['data']:
        strike = row.get('strikePrice')
        for t,opt in [('CE', row.get('CE')),('PE', row.get('PE'))]:
            if opt:
                df.append({'strike':strike,'type':t,'oi':opt.get('openInterest',0),
                           'oi_change':opt.get('changeinOpenInterest',0),
                           'lastPrice':opt.get('lastPrice',0),
                           'volume':opt.get('totalTradedVolume',0),
                           'expiry':opt.get('expiryDate')})
    df = pd.DataFrame(df)
    if df.empty: return []

    expiry = df['expiry'].value_counts().idxmax()
    df = df[df['expiry']==expiry]
    atm_strikes = pick_atm_strikes(sorted(df['strike'].unique()), spot)
    trend = trend_confirmations(symbol)
    if trend['bias']=="NEUTRAL": return []

    suggestions=[]
    for strike in atm_strikes:
        for t in ['CE','PE']:
            row=df[(df['strike']==strike)&(df['type']==t)]
            if row.empty: continue
            row=row.iloc[0]
            if row['oi']<MIN_OI or row['volume']<MIN_VOL or row['lastPrice']<=0: continue

            premium=row['lastPrice']
            target=round(premium*TARGET_MULTIPLIER,2)
            sl=round(premium*SL_MULTIPLIER,2)

            if STRICT_TREND_GATING:
                if trend['bias']=="BULLISH" and t!="CE": continue
                if trend['bias']=="BEARISH" and t!="PE": continue

            conf=0.6
            if row['oi']>=MIN_OI: conf+=0.1
            if row['oi_change']>0: conf+=0.1
            if row['volume']>2*MIN_VOL: conf+=0.05
            confidence=round(min(1.0,conf),2)

            suggestions.append({'symbol':symbol,'strike':strike,'type':t,'expiry':row['expiry'],
                                'premium':premium,'target':target,'sl':sl,'oi':row['oi'],
                                'oi_change':row['oi_change'],'volume':row['volume'],
                                'confidence':confidence,'spot':spot,
                                'rsi':trend['rsi'],'macd':trend['macd'],'macd_signal':trend['macd_signal']})
    return sorted(suggestions,key=lambda x:(x['confidence'],x['oi']),reverse=True)

def check_performance():
    # Ensure history file exists even if no active suggestions
    if not os.path.exists(HISTORY_FILE):
        pd.DataFrame(columns=['symbol', 'type', 'strike', 'premium', 'target', 'sl', 'current', 'status', 'action']).to_csv(HISTORY_FILE, index=False)
    
    if not os.path.exists(ACTIVE_FILE) or os.path.getsize(ACTIVE_FILE) == 0:
        print("[i] No active suggestions found.")
        return
    
    df=pd.read_csv(ACTIVE_FILE)
    updates=[]
    report=""
    for idx,row in df.iterrows():
        sym=row['symbol']
        typ=row['type']
        strike=row['strike']
        try:
            data = nse_optionchain_scrapper(sym)
            df_opt=[]
            for r in data['records']['data']:
                s=r.get('strikePrice')
                opt=r.get(typ)
                if opt and s==strike: df_opt.append(opt)
            if not df_opt: continue
            current_price = df_opt[0]['lastPrice']
        except: 
            current_price=None

        status="OPEN"
        action="HOLD"
        if current_price is not None:
            if current_price>=row['target']: 
                status="TARGET HIT"
                action="EXIT"
            elif current_price<=row['sl']: 
                status="SL HIT"
                action="EXIT"

        updates.append({
            'symbol': sym,
            'type': typ,
            'strike': strike,
            'premium': row['premium'],
            'target': row['target'],
            'sl': row['sl'],
            'current': current_price,
            'status': status,
            'action': action
        })
        report += f"{sym} {typ} {strike} premium:{row['premium']} current:{current_price} status:{status} action:{action}\n"

    if report:
        telegram_send("📊 Performance Update:\n"+report)
    
    # append to history CSV
    if os.path.exists(HISTORY_FILE):
        hist = pd.read_csv(HISTORY_FILE)
    else:
        hist = pd.DataFrame()
        
    if updates:
        hist = pd.concat([hist, pd.DataFrame(updates)], ignore_index=True)
        hist.to_csv(HISTORY_FILE, index=False)
    
    print("[i] Performance check completed.")

def main():
    mode = os.environ.get("MODE","manual")
    print(f"[i] Running in MODE={mode}")

    if mode=="suggestions":
        symbols = get_nifty50_list()
        all_suggestions=[]
        for i,sym in enumerate(symbols):
            print(f"[i] {i+1}/{len(symbols)} {sym}")
            try:
                sgs=analyze_option_chain(sym)
                if sgs: all_suggestions.extend(sgs)
            except:
                traceback.print_exc()
            time.sleep(REQUEST_DELAY)

        if not all_suggestions:
            print("[i] No strong suggestions found.")
            # Create empty active file if no suggestions
            pd.DataFrame(columns=['symbol', 'strike', 'type', 'expiry', 'premium', 'target', 'sl', 'oi', 'oi_change', 'volume', 'confidence', 'spot', 'rsi', 'macd', 'macd_signal']).to_csv(ACTIVE_FILE, index=False)
            return

        df=pd.DataFrame(all_suggestions).sort_values(['confidence','oi'],ascending=False)
        top=df[df['confidence']>=DF_MIN_CONF].head(5)
        if top.empty: top=df.head(3)
        top.to_csv(ACTIVE_FILE,index=False)
        msg="💡 New Suggestions:\n"
        for idx,row in top.iterrows():
            msg+=f"{row['symbol']} {row['type']} {row['strike']} premium:{row['premium']} target:{row['target']} sl:{row['sl']}\n"
        telegram_send(msg)
        print("[i] Suggestions generated and sent.")

    elif mode=="performance":
        check_performance()

    else:
        print("[i] Manual mode: nothing to do.")

if __name__=="__main__":
    main()
