# -*- coding: utf-8 -*-
"""Explore the tradingagentscn MongoDB for kline data usable by KiT testing."""
import os

from pymongo import MongoClient

URI = os.environ.get("MONGO_URI", "mongodb://127.0.0.1:27017/tradingagentscn")
c = MongoClient(URI, serverSelectionTimeoutMS=8000)
db = c["tradingagentscn"]

print("=== stock_daily_quotes ===")
col = db["stock_daily_quotes"]
print("docs:", col.estimated_document_count())
print("periods:", col.distinct("period"))
print("data_sources:", col.distinct("data_source"))
print("markets:", col.distinct("market"))
for period in col.distinct("period"):
    n = col.count_documents({"period": period})
    first = col.find_one({"period": period}, sort=[("trade_date", 1)])
    last = col.find_one({"period": period}, sort=[("trade_date", -1)])
    print(f"  period={period}: n={n}, range {first['trade_date']} .. {last['trade_date']}, "
          f"~{col.count_documents({'period': period, 'trade_date': last['trade_date']})} symbols on last day")
print("sample doc (daily):")
d = col.find_one({"period": "daily"})
print({k: d.get(k) for k in ("symbol", "full_symbol", "trade_date", "open", "high", "low", "close",
                             "pre_close", "volume", "amount", "adj_factor", "data_source")})

print("\n=== stock_minute_bars ===")
col2 = db["stock_minute_bars"]
print("docs:", col2.estimated_document_count())
first = col2.find_one(sort=[("datetime", 1)])
last = col2.find_one(sort=[("datetime", -1)])
print("range:", first["datetime"], "..", last["datetime"])
print("codes:", len(col2.distinct("code")))
print("sample doc:", {k: first.get(k) for k in ("code", "datetime", "open", "high", "low", "close", "volume")})

print("\n=== intraday_backtest_bars ===")
col3 = db["intraday_backtest_bars"]
print("docs:", col3.estimated_document_count())
if col3.estimated_document_count() > 0:
    f3 = col3.find_one(sort=[("datetime", 1)])
    l3 = col3.find_one(sort=[("datetime", -1)])
    print("range:", f3["datetime"], "..", l3["datetime"], "| codes:", len(col3.distinct("code")))
    print("sample doc:", {k: f3.get(k) for k in ("code", "datetime", "open", "high", "low", "close", "volume")})

print("\n=== trade_calendar ===")
col4 = db["trade_calendar"]
print("docs:", col4.estimated_document_count())
print("sample:", col4.find_one())

c.close()
