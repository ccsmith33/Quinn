"""Diagnostic probe — investigate why universe refresh produced 0 members.

Run with:
    bash -c 'set -a; source .env; set +a; .venv/bin/python ops/scripts/probe_universe.py'
"""
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetStatus
import os

client = TradingClient(
    api_key=os.environ["ALPACA_API_KEY_ID"],
    secret_key=os.environ["ALPACA_API_SECRET_KEY"],
    paper=True,
)

assets = client.get_all_assets(filter=GetAssetsRequest(status=AssetStatus.ACTIVE))
print(f"=== Alpaca ===")
print(f"total active: {len(assets)}")

us_eq = [a for a in assets if a.asset_class == "us_equity"]
print(f"us_equity: {len(us_eq)}")
print(f"sample symbols: {[a.symbol for a in us_eq[:5]]}")
print(f"exchanges: {sorted(set(str(a.exchange) for a in us_eq))[:15]}")
print(f"sample tradable=True: {len([a for a in us_eq if getattr(a, 'tradable', True)])}")

print(f"\n=== SEC ===")
from universe.sec_tickers import fetch_sec_tickers
secs = fetch_sec_tickers(user_agent="Quinn-Research/v1 ccsmith33@crimson.ua.edu")
print(f"sec tickers: {len(secs)}")
print(f"sample: {[s.ticker for s in secs[:5]]}")

print(f"\n=== Intersection ===")
sec_tickers = set(s.ticker.upper() for s in secs)
alpaca_tickers = set(a.symbol.upper() for a in us_eq)
overlap = sec_tickers & alpaca_tickers
print(f"sec tickers: {len(sec_tickers)}")
print(f"alpaca us_equity tickers: {len(alpaca_tickers)}")
print(f"intersection: {len(overlap)}")
print(f"sample overlap: {sorted(overlap)[:10]}")

print(f"\n=== yfinance probe (3 random tickers) ===")
import random
random.seed(42)
sample = random.sample(sorted(overlap), min(3, len(overlap)))
from universe.yfinance_provider import YFinanceProvider
yf = YFinanceProvider()
for ticker in sample:
    f = yf.fetch_fundamentals(ticker)
    print(f"  {ticker}: {f}")
