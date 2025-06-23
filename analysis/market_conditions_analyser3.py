# Required Libraries:
# pip install oandapyV20 pandas numpy ta-lib scikit-learn requests

# Auto-generate requirements.txt if it doesn't exist
import os

required_packages = [
    "oandapyV20",
    "pandas",
    "numpy",
    "ta-lib",
    "scikit-learn",
    "requests"
]

if not os.path.exists("requirements.txt"):
    with open("requirements.txt", "w") as f:
        f.write("\n".join(required_packages))
    print("✅ requirements.txt created successfully.")

import oandapyV20
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.instruments as instruments
import pandas as pd
import numpy as np
import time
import talib as ta
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import logging
from itertools import combinations

# ---- OANDA API Credentials ----
OANDA_API_KEY = "f7eff581944bb0b5efb4cac08003be9d-feea72696d43fb03101ddaa84eea2148"
OANDA_ACCOUNT_ID = "101-004-1683826-005"
OANDA_INSTRUMENTS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD",
    "USD_CHF", "NZD_USD", "EUR_GBP", "EUR_JPY", "GBP_JPY",
    "AUD_JPY", "EUR_AUD", "EUR_CAD", "AUD_CAD", "GBP_CAD",
    "NZD_JPY", "USD_SGD", "USD_HKD", "USD_MXN", "USD_ZAR",
    "USD_NOK", "USD_SEK", "EUR_NOK", "EUR_SEK", "EUR_NZD",
    "GBP_AUD", "GBP_NZD", "GBP_CHF", "NZD_CAD", "NZD_CHF",
    "EUR_CHF", "AUD_NZD"
]
TIMEFRAMES = ["M30", "H1", "D"]  # 30-min, 1-hour, daily

# Initialize OANDA API client
client = oandapyV20.API(access_token=OANDA_API_KEY)

# Logging setup
logging.basicConfig(filename='market_analysis.log', level=logging.INFO, format='%(asctime)s - %(message)s')

def fetch_real_time_prices():
    """
    Fetch real-time bid and ask prices for all configured instruments.
    
    This function retrieves current market prices from the OANDA API
    for all instruments in the OANDA_INSTRUMENTS list.
    
    Returns:
        dict: Dictionary mapping instrument symbols to (bid_price, ask_price) tuples,
              or empty dict if API call fails
    """
    try:
        params = {"instruments": ",".join(OANDA_INSTRUMENTS)}
        r = pricing.PricingInfo(accountID=OANDA_ACCOUNT_ID, params=params)
        client.request(r)
        return {
            p["instrument"]: (
                float(p["bids"][0]["price"]),
                float(p["asks"][0]["price"])
            ) for p in r.response["prices"]
        }
    except Exception as e:
        logging.error(f"Error fetching real-time prices: {e}")
        return {}

def fetch_historical_data(instrument, granularity):
    """
    Fetch historical OHLC data for a specific instrument and timeframe.
    
    This function retrieves 500 candlestick records with specified granularity
    from the OANDA API for the specified instrument.
    
    Args:
        instrument (str): The trading instrument symbol (e.g., 'EUR_USD')
        granularity (str): Timeframe granularity (e.g., 'M30', 'H1', 'D')
        
    Returns:
        pandas.DataFrame: DataFrame with columns ['Open', 'High', 'Low', 'Close'],
                         or empty DataFrame if API call fails
    """
    try:
        params = {"count": 500, "granularity": granularity}
        r = instruments.InstrumentsCandles(instrument=instrument, params=params)
        client.request(r)
        candles = r.response.get("candles", [])
        data = [
            {"Open": float(c["mid"]["o"]), "High": float(c["mid"]["h"]),
             "Low": float(c["mid"]["l"]), "Close": float(c["mid"]["c"])}
            for c in candles if "mid" in c
        ]
        return pd.DataFrame(data)
    except Exception as e:
        logging.error(f"Error fetching historical {granularity} for {instrument}: {e}")
        return pd.DataFrame()

def compute_indicators(df):
    """
    Compute technical indicators for the given price data.
    
    This function calculates various technical indicators including RSI, ADX,
    MACD, EMAs, and ATR. It handles missing data by forward and backward filling.
    
    Args:
        df (pandas.DataFrame): DataFrame with OHLC price data
        
    Returns:
        pandas.DataFrame: DataFrame with original data plus computed indicators
    """
    if df.empty:
        return df
    df['RSI'] = ta.RSI(df['Close'], 14)
    df['ADX'] = ta.ADX(df['High'], df['Low'], df['Close'], 14)
    df['MACD'], df['MACD_Signal'], _ = ta.MACD(df['Close'], 12, 26, 9)
    df['EMA_20'] = ta.EMA(df['Close'], 20)
    df['EMA_50'] = ta.EMA(df['Close'], 50)
    df['EMA_200'] = ta.EMA(df['Close'], 200)
    df['ATR'] = ta.ATR(df['High'], df['Low'], df['Close'], 14)
    df.bfill(inplace=True); df.ffill(inplace=True)
    return df

def evaluate_feature_combinations(df):
    """
    Evaluate different combinations of technical indicators for price prediction.
    
    This function tests various combinations of technical indicators to find
    the best feature set for price prediction using linear regression.
    
    Args:
        df (pandas.DataFrame): DataFrame with price data and technical indicators
        
    Returns:
        tuple: (best_feature_combination, best_metrics_dict)
    """
    features = ["RSI", "ADX", "MACD", "MACD_Signal", "EMA_20", "EMA_50", "EMA_200", "ATR"]
    df = df.dropna(subset=features + ['Close'])
    best_score, best_combo, best_metrics = -np.inf, [], {}
    for r in range(2, len(features)+1):
        for combo in combinations(features, r):
            try:
                X, y = df[list(combo)], df['Close']
                X_tr, X_te, y_tr, y_te = train_test_split(X, y, shuffle=False, test_size=0.2)
                mdl = LinearRegression().fit(X_tr, y_tr)
                pred = mdl.predict(X_te)
                r2 = r2_score(y_te, pred)
                mae = mean_absolute_error(y_te, pred)
                rmse = np.sqrt(mean_squared_error(y_te, pred))
                if r2 > best_score:
                    best_score = r2; best_combo = combo
                    best_metrics = {"R2": r2, "MAE": mae, "RMSE": rmse}
            except:
                continue
    return best_combo, best_metrics

def predict_price_with_best_features(df, features):
    """
    Predict price using the best feature combination.
    
    This function uses linear regression with the best feature combination
    to predict the next price value.
    
    Args:
        df (pandas.DataFrame): DataFrame with price data and indicators
        features (list): List of feature names to use for prediction
        
    Returns:
        float: Predicted price for the next period
    """
    features = list(features)
    X = df[features].dropna()
    y = df.loc[X.index, 'Close']
    mdl = LinearRegression().fit(X, y)
    next_feat = df[features].iloc[-1:].copy()
    return mdl.predict(next_feat)[0]

def detect_anomalies(df):
    """
    Detect price anomalies based on ATR volatility.
    
    This function identifies anomalous price movements by comparing
    the current ATR to its 20-period rolling mean.
    
    Args:
        df (pandas.DataFrame): DataFrame with price data and ATR indicator
        
    Returns:
        str: "Anomaly Detected" or "No Anomalies Detected"
    """
    return "Anomaly Detected" if df['ATR'].iloc[-1] > df['ATR'].rolling(20).mean().iloc[-1] *2 else "No Anomalies Detected"

def detect_trend_signal(df):
    """
    Detect trend strength using ADX indicator.
    
    This function determines if the market is trending strongly
    based on the ADX (Average Directional Index) value.
    
    Args:
        df (pandas.DataFrame): DataFrame with price data and ADX indicator
        
    Returns:
        str: "Strong Trend" if ADX > 25, otherwise "Weak Trend (ADX Below 25)"
    """
    return "Strong Trend" if df['ADX'].iloc[-1] > 25 else "Weak Trend (ADX Below 25)"

def detect_momentum_signal(df):
    """
    Detect momentum direction using MACD indicator.
    
    This function determines the momentum direction by comparing
    the MACD line to its signal line.
    
    Args:
        df (pandas.DataFrame): DataFrame with price data and MACD indicators
        
    Returns:
        str: Momentum signal indicating long, short, or no momentum
    """
    m, s = df['MACD'].iloc[-1], df['MACD_Signal'].iloc[-1]
    return "Momentum: Strong Upward (Long Position)" if m > s else "Momentum: Strong Downward (Short Position)" if m < s else "No Momentum"

def detect_macd_signal(df):
    """
    Detect MACD crossover signals.
    
    This function identifies bullish or bearish MACD signals
    based on the relationship between MACD and signal lines.
    
    Args:
        df (pandas.DataFrame): DataFrame with price data and MACD indicators
        
    Returns:
        str: "Bullish MACD Signal", "Bearish MACD Signal", or "No Strong Trend"
    """
    m, s = df['MACD'].iloc[-1], df['MACD_Signal'].iloc[-1]
    return "Bullish MACD Signal" if m > s else "Bearish MACD Signal" if m < s else "No Strong Trend"

def detect_pinbar_signal(df):
    """
    Detect pinbar (hammer) candlestick patterns.
    
    This function identifies pinbar patterns using the CDLHAMMER
    function from the TA-Lib library.
    
    Args:
        df (pandas.DataFrame): DataFrame with OHLC price data
        
    Returns:
        str: "Pinbar Pattern Detected" or "No Pinbar Pattern"
    """
    return "Pinbar Pattern Detected" if ta.CDLHAMMER(df['Open'], df['High'], df['Low'], df['Close']).iloc[-1] != 0 else "No Pinbar Pattern"

def detect_spread_widening(bid, ask):
    """
    Detect if the bid-ask spread is unusually wide.
    
    This function compares the current spread to a threshold
    to identify potential market stress or low liquidity.
    
    Args:
        bid (float): Current bid price
        ask (float): Current ask price
        
    Returns:
        str: "Widened Spread Detected" if spread > 0.01, otherwise "Normal Spread"
    """
    return "Widened Spread Detected" if (ask - bid) > 0.01 else "Normal Spread"

def detect_volatility_clustering(df):
    """
    Detect high volatility clustering periods.
    
    This function identifies periods of unusually high volatility
    by comparing current ATR to its historical average.
    
    Args:
        df (pandas.DataFrame): DataFrame with price data and ATR indicator
        
    Returns:
        str: "High Volatility Cluster Detected" or "Normal Volatility"
    """
    return "High Volatility Cluster Detected" if df['ATR'].iloc[-1] > df['ATR'].rolling(20).mean().iloc[-1] *2 else "Normal Volatility"

def detect_breakout(df):
    """
    Detect price breakouts above recent highs.
    
    This function identifies when the current high exceeds
    the 20-period rolling maximum high.
    
    Args:
        df (pandas.DataFrame): DataFrame with price data
        
    Returns:
        str: "Breakout Detected" or "No Breakout"
    """
    return "Breakout Detected" if df['High'].iloc[-1] > df['High'].rolling(20).max().iloc[-2] else "No Breakout"

def evaluate_market_condition(df, prediction):
    """
    Evaluate overall market condition and suggest strategy.
    
    This function compares the predicted price to the current close
    to determine market direction and suggest appropriate trading strategy.
    
    Args:
        df (pandas.DataFrame): DataFrame with price data
        prediction (float): Predicted price for next period
        
    Returns:
        tuple: (market_condition, recommended_strategy)
    """
    last = df['Close'].iloc[-1]
    if prediction is None: return "No Prediction", "Safe Default Strategy"
    return ("Trending Up", "Buy Strategy") if prediction > last else ("Trending Down", "Sell Strategy") if prediction < last else ("Sideways Market", "Neutral Strategy")

def calculate_strategy_confidence(signals, strategy: str):
    """
    Calculate confidence level for a trading strategy.
    
    This function evaluates how well the detected signals align
    with the recommended strategy to calculate a confidence percentage.
    
    Args:
        signals (dict): Dictionary containing various market signals
        strategy (str): The recommended trading strategy
        
    Returns:
        float: Confidence percentage (0-100)
    """
    score = 0; total=4
    if strategy=="Sell Strategy":
        score += signals["momentum"].endswith("Short Position")
        score += signals["macd"]=="Bearish MACD Signal"
        score += "Trend" in signals["trend"]
        score += signals["market_condition"]=="Trending Down"
    elif strategy=="Buy Strategy":
        score += signals["momentum"].endswith("Long Position")
        score += signals["macd"]=="Bullish MACD Signal"
        score += "Trend" in signals["trend"]
        score += signals["market_condition"]=="Trending Up"
    return round(score/total*100,2)

def final_strategy_decision(all_signals):
    """
    Make final strategy decision based on weighted signal analysis.
    
    This function aggregates signals from all instruments and timeframes
    to make a comprehensive strategy decision using weighted scoring.
    
    Args:
        all_signals (list): List of signal lists from all analyses
        
    Returns:
        str: Final recommended strategy
    """
    weights = {
        "Strong Trend":3, "Weak Trend":1,
        "Momentum: Strong Upward (Long Position)":2,
        "Momentum: Strong Downward (Short Position)":2,
        "Bullish MACD Signal":2, "Bearish MACD Signal":2,
        "Breakout Detected":3, "High Volatility Cluster Detected":2,
        "Widened Spread Detected":1
    }
    totals = {"Buy Strategy":0,"Sell Strategy":0,"Neutral Strategy":0,"Scalping Strategy":0,"Caution Strategy":0}
    for sigs in all_signals:
        for s in sigs:
            wt = weights.get(s,0)
            if "Upward" in s or "Bullish" in s:
                totals["Buy Strategy"] += wt
            elif "Downward" in s or "Bearish" in s:
                totals["Sell Strategy"] += wt
            elif "Neutral" in s:
                totals["Neutral Strategy"] += wt
            elif "Scalping" in s:
                totals["Scalping Strategy"] += wt
            elif "Caution" in s:
                totals["Caution Strategy"] += wt
    return max(totals, key=totals.get)

def main():
    """
    Main function for continuous market analysis across multiple timeframes.
    
    This function runs an infinite loop that continuously analyzes
    all configured instruments across multiple timeframes, prints results,
    and logs findings. It runs every 60 seconds.
    """
    all_signals = []
    while True:
        price_data = fetch_real_time_prices()
        for inst in OANDA_INSTRUMENTS:
            if inst not in price_data:
                logging.error(f"Missing price for {inst}")
                continue
            bid, ask = price_data[inst]
            for tf in TIMEFRAMES:
                print(f"\n📊 Market analysis: {inst} [{tf}]")
                df = fetch_historical_data(inst, tf)
                if df.empty:
                    logging.error(f"[{tf}] no data for {inst}"); continue
                df = compute_indicators(df)
                best_feat, metrics = evaluate_feature_combinations(df)
                pred = predict_price_with_best_features(df, best_feat)
                anomaly = detect_anomalies(df)
                trend = detect_trend_signal(df)
                momentum = detect_momentum_signal(df)
                macd = detect_macd_signal(df)
                pinbar = detect_pinbar_signal(df)
                spread = detect_spread_widening(bid, ask)
                volatility = detect_volatility_clustering(df)
                breakout = detect_breakout(df)
                cond, strat = evaluate_market_condition(df, pred)
                sig_dict = {"momentum":momentum, "macd":macd, "trend":trend, "market_condition":cond}
                confidence = calculate_strategy_confidence(sig_dict, strat)
                all_signals.append([trend, momentum, macd, breakout, volatility, spread])

                print(f"Predicted Price:     {pred:.6f}")
                print(f"Best Features:       {best_feat}")
                print(f"Model Accuracy - R²: {metrics['R2']:.4f}, MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}")
                print(f"Trend:               {trend}")
                print(f"Momentum:            {momentum}")
                print(f"MACD:                {macd}")
                print(f"Breakout:            {breakout}")
                print(f"Volatility:          {volatility}")
                print(f"Pinbar:              {pinbar}")
                print(f"Spread:              {spread}")
                print(f"Anomaly:             {anomaly}")
                print(f"Market Condition:    {cond}")
                print(f"Recommended Strat:   {strat}")
                print(f"Strategy Confidence: {confidence}%")
                logging.info(f"[{tf}] {inst} -> Strat: {strat} ({confidence}% conf)")

        final_strat = final_strategy_decision(all_signals)
        print(f"\n✅ Final Combined Strategy: {final_strat}")
        all_signals.clear()
        time.sleep(60)

if __name__ == "__main__":
    main()