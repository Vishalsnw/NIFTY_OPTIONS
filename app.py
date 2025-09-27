"""
Updated professional option suggestion script
Requirements:
pip install nsepython pandas numpy yfinance requests
"""
import pandas as pd
import numpy as np
import time, traceback, os, math
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
MIN_OI = 3000           # lowered to consider more strikes (tune as needed)
MIN_VOL = 200
STRIKES_AROUND_ATM = 2
TARGET_MULTIPLIER = 1.6
SL_MULTIPLIER = 0.6
REQUEST_DELAY = 1

RSI_PERIOD = 14
RSI_THRESHOLD = 50
STRICT_TREND_GATING = True
DF_MIN_CONF = 0.75

BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

ACTIVE_FILE = "active_suggestions.csv"
HISTORY_FILE = "all_history.csv"

RISK_FREE_RATE = 0.06   # annual risk free (adjust if needed)
IV_SELL_THRESHOLD = 0.8 # IV rank above which selling strategies preferred
IV_BUY_THRESHOLD = 0.4  # IV rank below which buying strategies preferred
PCR_BULLISH = 1.1
PCR_BEARISH = 0.9
# ----------------------------

def telegram_send(message):
    if not BOT_TOKEN or not CHAT_ID:
        print("[!] Telegram BOT_TOKEN or CHAT_ID missing")
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        requests.get(url, params={"chat_id":CHAT_ID,"text":message}, timeout=10)
    except Exception as e:
        print(f"[!] Telegram send failed: {e}")

def get_nifty50_list():
    try:
        url = "https://archives.nseindia.com/content/indices/ind_nifty50list.csv"
        df = pd.read_csv(url)
        return df['Symbol'].astype(str).str.strip().tolist()
    except:
        # Fallback list if NSE website is unavailable
        return ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "HDFC", "HINDUNILVR", 
                "KOTAKBANK", "BHARTIARTL", "ITC", "SBIN", "BAJFINANCE", "ASIANPAINT", 
                "MARUTI", "TITAN", "SUNPHARMA", "AXISBANK", "ULTRACEMCO", "TECHM", 
                "NESTLEIND", "TATAMOTORS", "WIPRO", "POWERGRID", "NTPC", "ONGC", 
                "ADANIPORTS", "JSWSTEEL", "HCLTECH", "DRREDDY", "HDFCLIFE", "BAJAJFINSV", 
                "CIPLA", "GRASIM", "TATASTEEL", "BRITANNIA", "COALINDIA", "INDUSINDBK", 
                "EICHERMOT", "BPCL", "HEROMOTOCO", "DIVISLAB", "SBILIFE", "SHREECEM", 
                "UPL", "APOLLOHOSP", "BAJAJ-AUTO", "M&M", "LT", "TATACONSUM"]

def pick_atm_strikes(strikes, spot):
    if not strikes:
        return []
    nearest = min(strikes, key=lambda x: abs(x - spot))
    idx = strikes.index(nearest)
    left = max(0, idx - STRIKES_AROUND_ATM)
    right = min(len(strikes)-1, idx + STRIKES_AROUND_ATM)
    return strikes[left:right+1]

# --- Technicals ---
def compute_rsi(series, period=14):
    if len(series) < period:
        return pd.Series([np.nan] * len(series))
    
    delta = series.diff().dropna()
    if len(delta) == 0:
        return pd.Series([np.nan] * len(series))
        
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_sma(series, window):
    if len(series) < window:
        return pd.Series([np.nan] * len(series))
    return series.rolling(window=window).mean()

def compute_macd(series, fast=12, slow=26, signal=9):
    if len(series) < slow:
        return pd.Series([np.nan] * len(series)), pd.Series([np.nan] * len(series))
        
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
        except Exception as e:
            print(f"[!] yfinance error for {symbol}: {e}")
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

    try:
        # Ensure we have at least 2 days of data
        if len(closes) < 2:
            return res
            
        yclose = closes.iloc[-2] if len(closes) >= 2 else closes.iloc[-1]
        lclose = closes.iloc[-1]
        res['price_gt_yesterday'] = (lclose > yclose)
        
        rsi_series = compute_rsi(closes, RSI_PERIOD)
        if not rsi_series.isna().all():
            rsi_val = rsi_series.iloc[-1]
            res['rsi'] = round(rsi_val, 2) if not pd.isna(rsi_val) else None
            res['rsi_ok'] = rsi_val > RSI_THRESHOLD if not pd.isna(rsi_val) else None
        
        sma20 = compute_sma(closes, 20).iloc[-1] if len(closes) >= 20 else None
        sma50 = compute_sma(closes, 50).iloc[-1] if len(closes) >= 50 else None
        
        if sma20 is not None and sma50 is not None and not pd.isna(sma20) and not pd.isna(sma50):
            res['sma20_gt_sma50'] = sma20 > sma50
        else:
            res['sma20_gt_sma50'] = None
            
        macd, macd_signal = compute_macd(closes)
        if not macd.isna().all() and not macd_signal.isna().all():
            macd_val = macd.iloc[-1]
            macd_sig_val = macd_signal.iloc[-1]
            res['macd'] = round(macd_val, 4) if not pd.isna(macd_val) else None
            res['macd_signal'] = round(macd_sig_val, 4) if not pd.isna(macd_sig_val) else None
            res['macd_ok'] = (macd_val > macd_sig_val) if not pd.isna(macd_val) and not pd.isna(macd_sig_val) else None

        # Determine bias only if we have sufficient data
        if (res['price_gt_yesterday'] is not None and res['rsi_ok'] is not None and 
            res['sma20_gt_sma50'] is not None and res['macd_ok'] is not None):
            
            bullish = (res['price_gt_yesterday'] and res['rsi_ok'] and 
                      res['sma20_gt_sma50'] and res['macd_ok'])
            bearish = ((not res['price_gt_yesterday']) and 
                      (res['rsi'] is not None and res['rsi'] < (100 - RSI_THRESHOLD)) and
                      (not res['sma20_gt_sma50']) and (not res['macd_ok']))

            if bullish:
                res['bias'] = "BULLISH"
            elif bearish:
                res['bias'] = "BEARISH"
            else:
                res['bias'] = "NEUTRAL"
        else:
            res['bias'] = "NEUTRAL"

    except Exception as e:
        print(f"[!] Error in trend analysis for {symbol}: {e}")
        res['bias'] = "NEUTRAL"

    return res

# --- Black Scholes / IV / Greeks ---
def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

def bs_price(spot, strike, t, vol, r, option_type):
    """Black Scholes price for European option (assuming no dividend). t in years, vol annual."""
    if t <= 0 or vol <= 0:
        # At expiry or zero vol
        if option_type == 'CE':
            return max(0.0, spot - strike)
        else:
            return max(0.0, strike - spot)
            
    try:
        d1 = (math.log(spot/strike) + (r + 0.5 * vol**2) * t) / (vol * math.sqrt(t))
        d2 = d1 - vol * math.sqrt(t)
        if option_type == 'CE':
            price = spot * _norm_cdf(d1) - strike * math.exp(-r*t) * _norm_cdf(d2)
        else:
            price = strike * math.exp(-r*t) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)
        return max(price, 0.0)  # Ensure non-negative price
    except (ValueError, ZeroDivisionError):
        return 0.0

def bs_greeks(spot, strike, t, vol, r, option_type):
    """Return delta, theta (per day), vega (per 1 vol point)."""
    if t <= 0 or vol <= 0:
        # At expiry, greeks are degenerate
        if option_type == 'CE':
            delta = 1.0 if spot > strike else 0.0
        else:
            delta = -1.0 if spot < strike else 0.0
        return {'delta': delta, 'theta': 0.0, 'vega': 0.0}

    try:
        d1 = (math.log(spot/strike) + (r + 0.5 * vol**2) * t) / (vol * math.sqrt(t))
        d2 = d1 - vol * math.sqrt(t)
        pdf_d1 = _norm_pdf(d1)
        
        if option_type == 'CE':
            delta = _norm_cdf(d1)
        else:
            delta = _norm_cdf(d1) - 1
            
        vega = spot * pdf_d1 * math.sqrt(t) / 100.0   # per 1 vol point (i.e., vol in decimal)
        
        # Theta approximate: use standard BS theta (per year) then divide by 365
        if option_type == 'CE':
            theta = (-spot * pdf_d1 * vol / (2 * math.sqrt(t)) - r * strike * math.exp(-r*t) * _norm_cdf(d2)) / 365.0
        else:
            theta = (-spot * pdf_d1 * vol / (2 * math.sqrt(t)) + r * strike * math.exp(-r*t) * _norm_cdf(-d2)) / 365.0
            
        return {'delta': delta, 'theta': theta, 'vega': vega}
    except (ValueError, ZeroDivisionError):
        return {'delta': 0.0, 'theta': 0.0, 'vega': 0.0}

def implied_vol_newton(mkt_price, spot, strike, t, r, option_type, initial=0.3, tol=1e-6, max_iter=80):
    """Implied vol via Newton-Raphson. Returns vol in decimal (e.g., 0.25)."""
    if mkt_price <= 0 or t <= 0:
        return None
        
    sigma = initial
    for i in range(max_iter):
        try:
            price = bs_price(spot, strike, t, sigma, r, option_type)
            diff = price - mkt_price
            if abs(diff) < tol:
                return max(sigma, 0.0)
                
            # compute vega (derivative wrt sigma)
            d1 = (math.log(spot/strike) + (r + 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))
            vega = spot * _norm_pdf(d1) * math.sqrt(t)
            if vega == 0:
                break
                
            sigma -= diff / vega
            if sigma <= 0:
                sigma = tol
        except (ValueError, ZeroDivisionError):
            break
            
    # fallback: if didn't converge, return None
    return None

# --- Chain metrics: PCR, IV Rank, Max Pain ---
def compute_chain_metrics(data, spot, expiry_date):
    """
    data: DataFrame with columns strike,type,oi,lastPrice,volume
    expiry_date: string 'DD-Mon-YYYY' or parseable
    """
    # time to expiry in years
    try:
        exp_dt = pd.to_datetime(expiry_date)
        days_to_expiry = max((exp_dt - datetime.now()).days, 0)
        t_years = days_to_expiry / 365.0
        if t_years <= 0:
            t_years = 1/365.0
    except:
        t_years = 7/365.0

    # Compute IV for each option (if possible)
    ivs = []
    for _, row in data.iterrows():
        mkt_price = row['lastPrice']
        strike = float(row['strike'])
        typ = row['type']
        
        # Avoid unrealistic tiny market prices
        if mkt_price is None or mkt_price <= 0.01:
            iv = None
        else:
            try:
                iv = implied_vol_newton(mkt_price, spot, strike, t_years, RISK_FREE_RATE, typ, initial=0.3)
            except Exception:
                iv = None
        ivs.append(iv)
        
    data = data.copy()
    data['iv'] = ivs

    # IV rank relative to current expiry distribution (best-effort)
    iv_vals = [iv for iv in ivs if iv is not None]
    if not iv_vals:
        iv_min = iv_max = np.nan
    else:
        iv_min = min(iv_vals)
        iv_max = max(iv_vals)
        
    def iv_rank_func(iv):
        try:
            if iv is None or np.isnan(iv_min) or np.isnan(iv_max) or iv_max == iv_min:
                return None
            return (iv - iv_min) / (iv_max - iv_min)
        except:
            return None
            
    data['iv_rank'] = data['iv'].apply(iv_rank_func)

    # PCR (OI-based) for the expiry: total PE OI / total CE OI
    total_pe_oi = data[data['type']=='PE']['oi'].sum()
    total_ce_oi = data[data['type']=='CE']['oi'].sum()
    pcr = None
    if total_ce_oi > 0:
        pcr = total_pe_oi / total_ce_oi

    # Max pain calculation (for possible settlement levels = strikes)
    strikes = sorted(data['strike'].unique())
    pain_totals = {}
    for s in strikes:
        total_pain = 0.0
        # calls: payout to buyers = max(0, spot_at_expiry - K) * OI
        for _, r in data[data['type']=='CE'].iterrows():
            payout = max(0, s - r['strike']) * r['oi']
            total_pain += payout
        # puts: payout to buyers = max(0, K - spot_at_expiry) * OI
        for _, r in data[data['type']=='PE'].iterrows():
            payout = max(0, r['strike'] - s) * r['oi']
            total_pain += payout
        pain_totals[s] = total_pain
        
    max_pain = min(pain_totals, key=pain_totals.get) if pain_totals else None

    return data, {'pcr': pcr, 'iv_min': iv_min, 'iv_max': iv_max, 'max_pain': max_pain, 't_years': t_years}

# --- Analyze option chain with advanced logic ---
def analyze_option_chain(symbol):
    try:
        data = nse_optionchain_scrapper(symbol)
        spot = float(data['records']['underlyingValue'])
    except Exception as e:
        print(f"[!] Error fetching option chain for {symbol}: {e}")
        return []

    rows = []
    for row in data['records']['data']:
        strike = row.get('strikePrice')
        if strike is None:
            continue
            
        for t,opt in [('CE', row.get('CE')),('PE', row.get('PE'))]:
            if opt:
                rows.append({
                    'strike': float(strike),
                    'type': t,
                    'oi': int(opt.get('openInterest', 0) or 0),
                    'oi_change': int(opt.get('changeinOpenInterest', 0) or 0),
                    'lastPrice': float(opt.get('lastPrice', 0) or 0),
                    'volume': int(opt.get('totalTradedVolume', 0) or 0),
                    'expiry': opt.get('expiryDate')
                })
                
    if not rows:
        print(f"[!] No option data found for {symbol}")
        return []
        
    df = pd.DataFrame(rows)
    expiry = df['expiry'].value_counts().idxmax()
    df = df[df['expiry']==expiry]
    
    strikes = sorted(df['strike'].unique())
    if not strikes:
        return []
        
    atm_strikes = pick_atm_strikes(strikes, spot)
    trend = trend_confirmations(symbol)

    # compute chain metrics
    df_metrics, chain_meta = compute_chain_metrics(df, spot, expiry)
    pcr = chain_meta['pcr']
    max_pain = chain_meta['max_pain']
    t_years = chain_meta['t_years']

    print(f"[i] {symbol} spot={spot} expiry={expiry} PCR={pcr:.2f}" if pcr else f"[i] {symbol} spot={spot} expiry={expiry}")

    suggestions = []
    # iterate strikes near ATM
    for strike in atm_strikes:
        for t in ['CE','PE']:
            row = df_metrics[(df_metrics['strike']==strike) & (df_metrics['type']==t)]
            if row.empty: 
                continue
                
            row = row.iloc[0]
            # basic liquidity filters
            if row['oi'] < MIN_OI or row['volume'] < MIN_VOL or row['lastPrice'] <= 0:
                continue

            premium = row['lastPrice']
            iv = row.get('iv', None)
            iv_rank = row.get('iv_rank', None)
            oi = row['oi']
            oi_change = row['oi_change']
            volume = row['volume']

            target = round(premium * TARGET_MULTIPLIER, 2)
            sl = round(premium * SL_MULTIPLIER, 2)

            # trend gating
            if STRICT_TREND_GATING and trend['bias'] in ['BULLISH', 'BEARISH']:
                if trend['bias']=="BULLISH" and t!="CE":
                    continue
                if trend['bias']=="BEARISH" and t!="PE":
                    continue

            # greeks (approx) using computed IV if available, else fallback with 0.3
            use_vol = iv if iv and iv>0 else 0.3
            greeks = bs_greeks(spot, strike, t_years, use_vol, RISK_FREE_RATE, t)
            delta = greeks['delta']
            theta = greeks['theta']
            vega = greeks['vega']

            # confidence scoring - incorporate OI, OI change, volume, trend alignment, IV rank, PCR signal
            conf = 0.4
            # liquidity boost
            if oi >= MIN_OI: conf += 0.1
            if oi_change > 0: conf += 0.05
            if volume > 2 * MIN_VOL: conf += 0.05
            # trend alignment
            if trend['bias']=='BULLISH' and t=='CE': conf += 0.1
            if trend['bias']=='BEARISH' and t=='PE': conf += 0.1
            # IV preference: if IV rank low -> buying premium (long) favored; if high -> selling favored
            if iv_rank is not None:
                if iv_rank < IV_BUY_THRESHOLD and t in ('CE','PE'):
                    conf += 0.07
                if iv_rank > IV_SELL_THRESHOLD:
                    # selling side gets preference
                    conf += 0.07
            # PCR signal
            if pcr:
                if pcr > PCR_BULLISH and t == 'CE':
                    conf += 0.05
                if pcr < PCR_BEARISH and t == 'PE':
                    conf += 0.05

            # cap confidence
            confidence = round(min(1.0, conf), 2)

            # strategy recommendation logic
            strategy = "BUY_OPTION"
            reason = []
            # if IV rank is high -> consider SELL strategies (short premium) or non-directional
            if iv_rank is not None and iv_rank >= IV_SELL_THRESHOLD:
                # prefer selling if adequate liquidity and margin understanding
                if trend['bias'] == 'NEUTRAL' or (trend['bias']=='BULLISH' and t=='CE') or (trend['bias']=='BEARISH' and t=='PE'):
                    strategy = "SELL_OPTION/SELL_SPREAD"
                    reason.append("High IV rank -> selling premium")
            else:
                if iv_rank is not None and iv_rank <= IV_BUY_THRESHOLD:
                    strategy = "BUY_OPTION"
                    reason.append("Low IV rank -> buying premium")
                else:
                    # mid IV -> directional buy if trend strong else consider debit spreads
                    if trend['bias'] in ('BULLISH','BEARISH'):
                        strategy = "BUY_OPTION"
                        reason.append("Directional bias + neutral IV")
                    else:
                        strategy = "DEBIT_SPREAD/NEUTRAL_STRATEGY"
                        reason.append("Neutral trend -> consider spreads")

            # if max_pain near spot, prefer range strategies
            if max_pain and abs(max_pain - spot) < (0.02 * spot):
                reason.append("Max pain near spot -> range bound potential")

            suggestions.append({
                'symbol': symbol,
                'strike': strike,
                'type': t,
                'expiry': row['expiry'],
                'premium': premium,
                'target': target,
                'sl': sl,
                'oi': oi,
                'oi_change': oi_change,
                'volume': volume,
                'iv': round(iv,4) if iv else None,
                'iv_rank': round(iv_rank,3) if iv_rank else None,
                'delta': round(delta,3),
                'theta': round(theta,4),
                'vega': round(vega,4),
                'confidence': confidence,
                'strategy': strategy,
                'reasons': "; ".join(reason) if reason else "",
                'spot': spot,
                'rsi': trend.get('rsi'),
                'bias': trend.get('bias'),
                'pcr': round(pcr,3) if pcr else None,
                'max_pain': max_pain
            })

    print(f"[i] Found {len(suggestions)} suggestions for {symbol}")
    # sort by confidence, then by oi
    return sorted(suggestions, key=lambda x:(x['confidence'], x['oi']), reverse=True)

def get_current_option_price(symbol, strike, option_type):
    """Get current price for a specific option with better error handling"""
    try:
        print(f"[i] Fetching current price for {symbol} {strike} {option_type}")
        data = nse_optionchain_scrapper(symbol)
        strike_float = float(strike)
        for row in data['records']['data']:
            row_strike = row.get('strikePrice')
            if row_strike is None:
                continue
            if abs(float(row_strike) - strike_float) < 0.01:
                option_data = row.get(option_type)
                if option_data:
                    current_price = option_data.get('lastPrice', None)
                    if current_price is not None:
                        current_price = float(current_price)
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
    history_columns = ['timestamp', 'symbol', 'type', 'strike', 'premium', 'target', 'sl', 'current', 'status', 'action']
    
    if not os.path.exists(HISTORY_FILE):
        pd.DataFrame(columns=history_columns).to_csv(HISTORY_FILE, index=False)
        print("[i] Created new history file")

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
        current_price = get_current_option_price(symbol, strike, option_type)
        status = "OPEN"
        action = "HOLD"
        
        if current_price is not None:
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
                pct_change = ((current_price_float - premium_float) / premium_float) * 100
                status = f"OPEN ({pct_change:+.1f}%)"
        else:
            status = "PRICE UNAVAILABLE"
            current_price = "N/A"

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

        report += f"• {symbol} {option_type} {strike}\n"
        report += f"  Premium: {premium} | Current: {current_price}\n"
        report += f"  Target: {row['target']} | SL: {row['sl']}\n"
        report += f"  Status: {status} | Action: {action}\n\n"

    if updates:
        telegram_send(report)
        print("[i] Performance report sent via Telegram")
    else:
        print("[!] No updates to report")

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
            # Create empty active file with proper columns
            columns = ['symbol', 'strike', 'type', 'expiry', 'premium', 'target', 'sl',
                       'oi', 'oi_change', 'volume', 'confidence', 'spot', 'rsi',
                       'iv', 'iv_rank', 'delta', 'theta', 'vega', 'strategy', 'reasons', 'pcr', 'max_pain']
            pd.DataFrame(columns=columns).to_csv(ACTIVE_FILE, index=False)
            telegram_send("❌ No trading suggestions found today. Market conditions may not be favorable.")
            return

        df = pd.DataFrame(all_suggestions)
        print(f"[i] Total suggestions found: {len(df)}")

        df = df.sort_values(['confidence', 'oi'], ascending=[False, False])
        top = df[df['confidence'] >= DF_MIN_CONF].head(7)
        if top.empty:
            top = df.head(5)
            print("[i] No suggestions met minimum confidence, taking top 5")

        print(f"[i] Selected {len(top)} top suggestions")
        top.to_csv(ACTIVE_FILE, index=False)
        print("[i] Suggestions saved to active_suggestions.csv")

        # Prepare telegram message with strategy hints
        msg = "💡 New Trading Suggestions (pro):\n\n"
        for idx, row in top.iterrows():
            msg += f"📌 {row['symbol']} | {row['type']} {int(row['strike'])} | Exp: {row['expiry']}\n"
            msg += f"   Premium: ₹{row['premium']} | Target: ₹{row['target']} | SL: ₹{row['sl']}\n"
            msg += f"   Conf: {row['confidence']} | OI: {int(row['oi']):,} | IV: {row.get('iv')}\n"
            msg += f"   IV_Rank: {row.get('iv_rank')} | Delta: {row.get('delta')} | Strategy: {row.get('strategy')}\n"
            if row.get('reasons'):
                msg += f"   Note: {row.get('reasons')}\n"
            msg += "\n"
        msg += "⚠️ Use proper risk management. Scripts provide ideas; not financial advice."
        telegram_send(msg)
        print("[i] Suggestions sent via Telegram")

    elif mode == "performance":
        print("[i] Starting performance check...")
        check_performance()
    else:
        print("[i] Manual mode: nothing to do.")

if __name__ == "__main__":
    main()