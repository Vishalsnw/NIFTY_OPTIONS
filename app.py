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
    except Exception as e:
        print(f"[!] Error fetching option chain for {symbol}: {e}")
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
    if df.empty: 
        print(f"[!] No option data found for {symbol}")
        return []

    expiry = df['expiry'].value_counts().idxmax()
    df = df[df['expiry']==expiry]
    atm_strikes = pick_atm_strikes(sorted(df['strike'].unique()), spot)
    trend = trend_confirmations(symbol)
    
    if trend['bias'] == "NEUTRAL": 
        print(f"[!] {symbol} has neutral trend bias, skipping")
        return []

    suggestions=[]
    for strike in atm_strikes:
        for t in ['CE','PE']:
            row=df[(df['strike']==strike)&(df['type']==t)]
            if row.empty: 
                continue
            row=row.iloc[0]
            if row['oi']<MIN_OI or row['volume']<MIN_VOL or row['lastPrice']<=0: 
                continue

            premium=row['lastPrice']
            target=round(premium*TARGET_MULTIPLIER,2)
            sl=round(premium*SL_MULTIPLIER,2)

            if STRICT_TREND_GATING:
                if trend['bias']=="BULLISH" and t!="CE": 
                    continue
                if trend['bias']=="BEARISH" and t!="PE": 
                    continue

            conf=0.6
            if row['oi']>=MIN_OI: conf+=0.1
            if row['oi_change']>0: conf+=0.1
            if row['volume']>2*MIN_VOL: conf+=0.05
            confidence=round(min(1.0,conf),2)

            suggestions.append({
                'symbol':symbol,'strike':strike,'type':t,'expiry':row['expiry'],
                'premium':premium,'target':target,'sl':sl,'oi':row['oi'],
                'oi_change':row['oi_change'],'volume':row['volume'],
                'confidence':confidence,'spot':spot,
                'rsi':trend['rsi'],'macd':trend['macd'],'macd_signal':trend['macd_signal']
            })
    
    print(f"[i] Found {len(suggestions)} suggestions for {symbol}")
    return sorted(suggestions,key=lambda x:(x['confidence'],x['oi']),reverse=True)

def get_current_option_price(symbol, strike, option_type):
    """Get current price for a specific option with better error handling"""
    try:
        print(f"[i] Fetching current price for {symbol} {strike} {option_type}")
        data = nse_optionchain_scrapper(symbol)
        
        # Convert strike to float for comparison
        strike_float = float(strike)
        
        for row in data['records']['data']:
            row_strike = row.get('strikePrice')
            if row_strike is None:
                continue
                
            # Compare strikes with tolerance for floating point issues
            if abs(float(row_strike) - strike_float) < 0.01:
                option_data = row.get(option_type)
                if option_data:
                    current_price = option_data.get('lastPrice', None)
                    print(f"[+] Found current price: {current_price}")
                    return current_price
        
        print(f"[!] Could not find option {symbol} {strike} {option_type} in current data")
        return None
        
    except Exception as e:
        print(f"[!] Error fetching current price for {symbol} {strike} {option_type}: {e}")
        return None

def check_performance():
    """Check performance of active suggestions with improved logic"""
    print("[i] Starting performance check...")
    
    # Ensure history file exists with proper columns
    history_columns = ['timestamp', 'symbol', 'type', 'strike', 'premium', 'target', 'sl', 'current', 'status', 'action']
    if not os.path.exists(HISTORY_FILE):
        pd.DataFrame(columns=history_columns).to_csv(HISTORY_FILE, index=False)
        print("[i] Created new history file")
    
    # Check if active suggestions file exists and has data
    if not os.path.exists(ACTIVE_FILE):
        print("[!] Active suggestions file not found")
        telegram_send("📊 Performance Update: No active suggestions file found.")
        return
    
    try:
        df = pd.read_csv(ACTIVE_FILE)
        if df.empty:
            print("[!] Active suggestions file is empty")
            telegram_send("📊 Performance Update: No active suggestions to track.")
            return
    except Exception as e:
        print(f"[!] Error reading active suggestions file: {e}")
        return
    
    print(f"[i] Found {len(df)} active suggestions to check")
    
    updates = []
    report = "📊 Performance Update:\n\n"
    
    for idx, row in df.iterrows():
        symbol = row['symbol']
        option_type = row['type']
        strike = row['strike']
        premium = row['premium']
        
        print(f"[i] Checking {symbol} {option_type} {strike}...")
        
        # Get current price
        current_price = get_current_option_price(symbol, strike, option_type)
        
        status = "OPEN"
        action = "HOLD"
        
        if current_price is not None:
            # Convert to float for comparison
            current_price_float = float(current_price)
            target_float = float(row['target'])
            sl_float = float(row['sl'])
            premium_float = float(premium)
            
            if current_price_float >= target_float: 
                status = "TARGET HIT 🎯"
                action = "EXIT ✅"
            elif current_price_float <= sl_float: 
                status = "SL HIT ⚠️"
                action = "EXIT ❌"
            else:
                # Calculate percentage change
                pct_change = ((current_price_float - premium_float) / premium_float) * 100
                status = f"OPEN ({pct_change:+.1f}%)"
        else:
            status = "PRICE UNAVAILABLE"
            current_price = "N/A"

        # Create update record
        update_record = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'symbol': symbol,
            'type': option_type,
            'strike': strike,
            'premium': premium,
            'target': row['target'],
            'sl': row['sl'],
            'current': current_price,
            'status': status,
            'action': action
        }
        updates.append(update_record)
        
        # Add to report
        report += f"• {symbol} {option_type} {strike}\n"
        report += f"  Premium: {premium} | Current: {current_price}\n"
        report += f"  Target: {row['target']} | SL: {row['sl']}\n"
        report += f"  Status: {status} | Action: {action}\n\n"

    # Send performance report
    if updates:
        telegram_send(report)
        print("[i] Performance report sent via Telegram")
    else:
        print("[!] No updates to report")
    
    # Update history file
    if updates:
        try:
            if os.path.exists(HISTORY_FILE):
                hist = pd.read_csv(HISTORY_FILE)
            else:
                hist = pd.DataFrame(columns=history_columns)
            
            new_records = pd.DataFrame(updates)
            hist = pd.concat([hist, new_records], ignore_index=True)
            hist.to_csv(HISTORY_FILE, index=False)
            print(f"[i] Updated history with {len(updates)} records")
        except Exception as e:
            print(f"[!] Error updating history file: {e}")
    else:
        print("[i] No updates to save")
    
    print("[i] Performance check completed.")

def main():
    mode = os.environ.get("MODE", "manual")
    print(f"[i] Running in MODE={mode}")

    if mode == "suggestions":
        print("[i] Starting suggestions generation...")
        symbols = get_nifty50_list()
        all_suggestions = []
        
        print(f"[i] Analyzing {len(symbols)} symbols...")
        
        for i, sym in enumerate(symbols):
            print(f"[i] {i+1}/{len(symbols)} Analyzing {sym}")
            try:
                suggestions = analyze_option_chain(sym)
                if suggestions: 
                    all_suggestions.extend(suggestions)
                    print(f"[+] Found {len(suggestions)} suggestions for {sym}")
            except Exception as e:
                print(f"[!] Error analyzing {sym}: {e}")
                traceback.print_exc()
            
            time.sleep(REQUEST_DELAY)

        if not all_suggestions:
            print("[!] No suggestions found for any symbol")
            # Create empty file with proper columns
            columns = ['symbol', 'strike', 'type', 'expiry', 'premium', 'target', 'sl', 
                      'oi', 'oi_change', 'volume', 'confidence', 'spot', 'rsi', 'macd', 'macd_signal']
            pd.DataFrame(columns=columns).to_csv(ACTIVE_FILE, index=False)
            telegram_send("❌ No trading suggestions found today. Market conditions may not be favorable.")
            return

        # Create DataFrame and sort
        df = pd.DataFrame(all_suggestions)
        print(f"[i] Total suggestions found: {len(df)}")
        
        # Sort by confidence and OI
        df = df.sort_values(['confidence', 'oi'], ascending=[False, False])
        
        # Filter by minimum confidence
        top = df[df['confidence'] >= DF_MIN_CONF].head(5)
        if top.empty: 
            top = df.head(3)
            print("[i] No suggestions met minimum confidence, taking top 3")
        
        print(f"[i] Selected {len(top)} top suggestions")
        
        # Save to file
        top.to_csv(ACTIVE_FILE, index=False)
        print("[i] Suggestions saved to active_suggestions.csv")
        
        # Send Telegram message
        msg = "💡 New Trading Suggestions:\n\n"
        for idx, row in top.iterrows():
            msg += f"📈 {row['symbol']} {row['type']} {row['strike']}\n"
            msg += f"   Premium: ₹{row['premium']} | Target: ₹{row['target']} | SL: ₹{row['sl']}\n"
            msg += f"   Confidence: {row['confidence']} | OI: {row['oi']:,}\n"
            msg += f"   Expiry: {row['expiry']}\n\n"
        
        telegram_send(msg)
        print("[i] Suggestions sent via Telegram")

    elif mode == "performance":
        print("[i] Starting performance check...")
        check_performance()

    else:
        print("[i] Manual mode: nothing to do.")

if __name__ == "__main__":
    main()
