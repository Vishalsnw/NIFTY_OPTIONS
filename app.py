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

BOT_TOKEN = "8050429062:AAFPLG9NuPnkDjVZyLUeg35Tlg4ArKisLbQ"
CHAT_ID = "-1002573892631"

ACTIVE_FILE = "active_suggestions.csv"
HISTORY_FILE = "all_history.csv"
# ----------------------------

def telegram_send(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        requests.get(url, params={"chat_id":CHAT_ID,"text":message})
    except: pass

# --- Existing functions: get_nifty50_list, pick_atm_strikes, compute_rsi, compute_sma, compute_macd, fetch_history, trend_confirmations ---
# (Use previous versions from your code with scalar fixes and bias)

# Analyze option chain
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

# --- Check performance ---
def check_performance():
    if not os.path.exists(ACTIVE_FILE): return
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
        except: current_price=None

        status="OPEN"
        action="HOLD"
        if current_price is not None:
            if current_price>=row['target']: status="TARGET HIT"; action="EXIT"
            elif current_price<=row['sl']: status="SL HIT"; action="EXIT"
        updates.append({'symbol':sym,'type':typ,'strike':strike,'premium':row['premium'],
                        'target':row['target'],'sl':row['sl'],'current':current_price,
                        'status':status,'action':action})
        report+=f"{sym} {typ} {strike} premium:{row['premium']} current:{current_price} status:{status} action:{action}\n"

    telegram_send("📊 Performance Update:\n"+report)
    # append to history
    if os.path.exists(HISTORY_FILE):
        hist=pd.read_csv(HISTORY_FILE)
    else:
        hist=pd.DataFrame()
    hist=pd.concat([hist,pd.DataFrame(updates)],ignore_index=True)
    hist.to_csv(HISTORY_FILE,index=False)

# --- Main ---
def main():
    now=datetime.now()
    hour=now.hour+5+(now.minute/60)  # rough IST hour
    # Evening suggestion generation ~ 9 PM IST
    if 21<=hour<22:
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
            return

        df=pd.DataFrame(all_suggestions).sort_values(['confidence','oi'],ascending=False)
        top=df[df['confidence']>=DF_MIN_CONF].head(5)
        if top.empty: top=df.head(3)
        top.to_csv(ACTIVE_FILE,index=False)
        msg="💡 New Suggestions:\n"
        for idx,row in top.iterrows():
            msg+=f"{row['symbol']} {row['type']} {row['strike']} premium:{row['premium']} target:{row['target']} sl:{row['sl']}\n"
        telegram_send(msg)

    # Evening performance check ~ 6 PM IST
    elif 18<=hour<19:
        check_performance()

if __name__=="__main__":
    main()
