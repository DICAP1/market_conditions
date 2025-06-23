"""
Application configuration settings.

This module contains configuration constants for the market analysis application,
including OANDA API credentials, Redis connection settings, and trading parameters.
"""

OANDA_API_KEY = "f7eff581944bb0b5efb4cac08003be9d-feea72696d43fb03101ddaa84eea2148"
OANDA_ACCOUNT_ID = "101-004-1683826-005"
REDIS_URL = "redis://localhost:6379/0"

OANDA_INSTRUMENTS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD",
    "USD_CHF", "NZD_USD", "EUR_GBP", "EUR_JPY", "GBP_JPY"
]

TIMEFRAME = "M1"
