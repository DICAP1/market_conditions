import oandapyV20
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.instruments as instruments
import pandas as pd
import numpy as np
import time
from datetime import datetime
import talib as ta
from sklearn.linear_model import LinearRegression
import logging

# ---- OANDA API Credentials ----
OANDA_API_KEY = "f7eff581944bb0b5efb4cac08003be9d-feea72696d43fb03101ddaa84eea2148"
OANDA_ACCOUNT_ID = "101-004-1683826-005"
OANDA_INSTRUMENTS = [
    # Major Forex Pairs
    "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD",
    "USD_CHF", "NZD_USD",

    # Minor Forex Pairs
    "EUR_GBP", "EUR_JPY", "GBP_JPY", "AUD_JPY", "EUR_AUD",
    "EUR_CAD", "AUD_CAD", "GBP_CAD", "NZD_JPY", "USD_SGD",
    "USD_HKD", "USD_MXN", "USD_ZAR", "USD_NOK", "USD_SEK",
    "EUR_NOK", "EUR_SEK", "EUR_NZD", "GBP_AUD", "GBP_NZD",
    "GBP_CHF", "NZD_CAD", "NZD_CHF", "EUR_CHF", "AUD_NZD",



]

# Initialize OANDA API client
client = oandapyV20.API(access_token=OANDA_API_KEY)

# Set up logging
logging.basicConfig(filename='market_analysis.log', level=logging.INFO, format='%(asctime)s - %(message)s')


def fetch_real_time_prices():
    try:
        params = {"instruments": ",".join(OANDA_INSTRUMENTS)}
        r = pricing.PricingInfo(accountID=OANDA_ACCOUNT_ID, params=params)
        client.request(r)
        prices = r.response["prices"]
        return {price["instrument"]: (float(price["bids"][0]["price"]), float(price["asks"][0]["price"])) for price in
                prices}
    except Exception as e:
        logging.error(f"Error fetching real-time prices: {str(e)}")
        return {}


def fetch_historical_data(instrument):
    try:
        params = {"count": 500, "granularity": "M30"}
        r = instruments.InstrumentsCandles(instrument=instrument, params=params)
        client.request(r)
        candles = r.response.get("candles", [])

        if not candles:
            return pd.DataFrame()

        data = [{"Open": float(c["mid"]["o"]), "High": float(c["mid"]["h"]), "Low": float(c["mid"]["l"]),
                 "Close": float(c["mid"]["c"])} for c in candles if "mid" in c]
        return pd.DataFrame(data)
    except Exception as e:
        logging.error(f"Error fetching historical data for {instrument}: {str(e)}")
        return pd.DataFrame()


def compute_indicators(df):
    if df.empty:
        return df

    df['RSI'] = ta.RSI(df['Close'], timeperiod=14)
    df['ADX'] = ta.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = ta.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['EMA_20'] = ta.EMA(df['Close'], timeperiod=20)
    df['EMA_50'] = ta.EMA(df['Close'], timeperiod=50)
    df['EMA_200'] = ta.EMA(df['Close'], timeperiod=200)
    df['ATR'] = ta.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)

    df.bfill(inplace=True)
    df.ffill(inplace=True)

    return df


def predict_price(df):
    if df.empty or len(df) < 20:
        return None
    X = np.arange(len(df)).reshape(-1, 1)
    y = df["Close"].values
    model = LinearRegression()
    model.fit(X, y)
    return model.predict([[len(df) + 1]])[0]


def detect_anomalies(df):
    if df['ATR'].iloc[-1] > df['ATR'].rolling(20).mean().iloc[-1] * 2:
        return "Anomaly Detected"
    return "No Anomalies Detected"


def detect_trend_signal(df):
    if df['ADX'].iloc[-1] > 25:
        return "Strong Trend"
    return "Weak Trend (ADX Below 25)"


def detect_momentum_signal(df):
    if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1]:
        return "Momentum: Strong Upward (Long Position)"
    elif df['MACD'].iloc[-1] < df['MACD_Signal'].iloc[-1]:
        return "Momentum: Strong Downward (Short Position)"
    return "No Momentum"


def detect_macd_signal(df):
    if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1]:
        return "Bullish MACD Signal"
    elif df['MACD'].iloc[-1] < df['MACD_Signal'].iloc[-1]:
        return "Bearish MACD Signal"
    return "No Strong Trend"


def detect_pinbar_signal(df):
    pinbar = ta.CDLHAMMER(df['Open'], df['High'], df['Low'], df['Close'])
    if pinbar.iloc[-1] != 0:
        return "Pinbar Pattern Detected"
    return "No Pinbar Pattern"


def detect_spread_widening(bid_price, ask_price):
    spread = ask_price - bid_price
    if spread > 0.01:
        return "Widened Spread Detected"
    return "Normal Spread"


def detect_volatility_clustering(df):
    if df['ATR'].iloc[-1] > df['ATR'].rolling(20).mean().iloc[-1] * 2:
        return "High Volatility Cluster Detected"
    return "Normal Volatility"


def detect_breakout(df):
    if df['High'].iloc[-1] > df['High'].rolling(20).max().iloc[-2]:
        return "Breakout Detected"
    return "No Breakout"


def evaluate_market_condition(df, predicted_price):
    if predicted_price is None:
        return "No Prediction", "Safe Default Strategy"

    last_close = df['Close'].iloc[-1]
    if predicted_price > last_close:
        return "Trending Up", "Buy Strategy"
    elif predicted_price < last_close:
        return "Trending Down", "Sell Strategy"
    else:
        return "Sideways Market", "Neutral Strategy"


def calculate_strategy_confidence(signals: dict, strategy: str) -> float:
    alignment_score = 0
    total_considered = 4

    if strategy == "Sell Strategy":
        alignment_score += signals.get("momentum") == "Momentum: Strong Downward (Short Position)"
        alignment_score += signals.get("macd") == "Bearish MACD Signal"
        alignment_score += "Trend" in signals.get("trend", "")
        alignment_score += signals.get("market_condition") == "Trending Down"

    elif strategy == "Buy Strategy":
        alignment_score += signals.get("momentum") == "Momentum: Strong Upward (Long Position)"
        alignment_score += signals.get("macd") == "Bullish MACD Signal"
        alignment_score += "Trend" in signals.get("trend", "")
        alignment_score += signals.get("market_condition") == "Trending Up"

    confidence = round((alignment_score / total_considered) * 100, 2)
    return confidence


def detect_market_signals(df, bid_price, ask_price):
    anomaly_detection = detect_anomalies(df)
    trend_signal = detect_trend_signal(df)
    momentum_signal = detect_momentum_signal(df)
    macd_signal = detect_macd_signal(df)
    pinbar_signal = detect_pinbar_signal(df)
    spread_widening = detect_spread_widening(bid_price, ask_price)
    volatility_clustering = detect_volatility_clustering(df)
    breakout_detection = detect_breakout(df)

    return anomaly_detection, trend_signal, momentum_signal, macd_signal, pinbar_signal, spread_widening, \
        volatility_clustering, breakout_detection


def final_strategy_decision(all_signals):
    signal_weights = {
        "Strong Trend": 3,
        "Weak Trend": 1,
        "Momentum: Strong Upward (Long Position)": 2,
        "Momentum: Strong Downward (Short Position)": 2,
        "Bullish MACD Signal": 2,
        "Bearish MACD Signal": 2,
        "Breakout Detected": 3,
        "High Volatility Cluster Detected": 2,
        "Widened Spread Detected": 1
    }

    strategy_counts = {
        "Buy Strategy": 0,
        "Sell Strategy": 0,
        "Neutral Strategy": 0,
        "Scalping Strategy": 0,
        "Caution Strategy": 0
    }

    for signals in all_signals:
        for signal in signals:
            weight = signal_weights.get(signal, 0)
            if "Upward" in signal or "Bullish" in signal:
                strategy_counts["Buy Strategy"] += weight
            elif "Downward" in signal or "Bearish" in signal:
                strategy_counts["Sell Strategy"] += weight
            elif "Neutral" in signal:
                strategy_counts["Neutral Strategy"] += weight
            elif "Scalping" in signal:
                strategy_counts["Scalping Strategy"] += weight
            elif "Caution" in signal:
                strategy_counts["Caution Strategy"] += weight

    final_strategy = max(strategy_counts, key=strategy_counts.get)
    return final_strategy


def main():
    all_signals = []

    while True:
        for instrument in OANDA_INSTRUMENTS:
            print(f"\nFetching market analysis for {instrument}...")

            price_data = fetch_real_time_prices()
            if instrument not in price_data:
                logging.error(f"Error fetching price data for {instrument}.")
                continue

            bid_price, ask_price = price_data[instrument]

            df = fetch_historical_data(instrument)
            if df.empty:
                logging.error(f"No historical data for {instrument}.")
                continue
            df = compute_indicators(df)

            predicted_price = predict_price(df)

            anomaly_detection, trend_signal, momentum_signal, macd_signal, pinbar_signal, spread_widening, \
                volatility_clustering, breakout_detection = detect_market_signals(df, bid_price, ask_price)

            market_condition, strategy = evaluate_market_condition(df, predicted_price)

            signal_dict = {
                "momentum": momentum_signal,
                "macd": macd_signal,
                "trend": trend_signal,
                "market_condition": market_condition
            }
            confidence_score = calculate_strategy_confidence(signal_dict, strategy)

            all_signals.append([trend_signal, momentum_signal, macd_signal, breakout_detection, volatility_clustering,
                                spread_widening])

            print(f"Market Analysis for {instrument}:")
            print(f"Instrument: {instrument}")
            print(f"Predicted Price: {predicted_price}")
            print(f"Trend Signal: {trend_signal}")
            print(f"Momentum Signal: {momentum_signal}")
            print(f"MACD Signal: {macd_signal}")
            print(f"Breakout Detection: {breakout_detection}")
            print(f"Volatility Clustering: {volatility_clustering}")
            print(f"Pinbar Signal: {pinbar_signal}")
            print(f"Spread Widening: {spread_widening}")
            print(f"Anomaly Detection: {anomaly_detection}")
            print(f"Market Condition: {market_condition}")
            print(f"Recommended Strategy: {strategy}")
            print(f"Strategy Confidence: {confidence_score}%")

            logging.info(f"{instrument} - Strategy: {strategy} (Confidence: {confidence_score}%)")

        final_strategy = final_strategy_decision(all_signals)
        print(f"\nFinal Strategy: {final_strategy}")

        all_signals.clear()
        time.sleep(60)


if __name__ == "__main__":
    main()
