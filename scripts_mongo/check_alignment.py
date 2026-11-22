# -*- coding: utf-8 -*-
"""Cross-check MongoDB daily bars vs KiT demo parquet: adjustment, volume unit, source consistency."""
import json
import os

import pandas as pd
from pymongo import MongoClient

URI = os.environ.get("MONGO_URI", "mongodb://127.0.0.1:27017/tradingagentscn")
db = MongoClient(URI, serverSelectionTimeoutMS=8000)["tradingagentscn"]

instruments = json.loads(open("data/instruments.json", encoding="utf-8").read())
demo_codes = list(instruments)          # e.g. "300368.SZ"
print("demo instruments:", len(demo_codes), "first:", demo_codes[:5])

col = db["stock_daily_quotes"]

# 1) which demo stocks exist per data_source
for src in ("tushare", "akshare", "baostock"):
    hits = 0
    for full in demo_codes:
        sym = full.split(".")[0]
        if col.find_one({"symbol": sym, "data_source": src, "period": "daily"}, {"_id": 1}):
            hits += 1
    print(f"overlap with demo 100 in source={src}: {hits}")

# 2) date-range + source counts for one representative stock
sym = "300368"
print(f"\n--- {sym} source/date breakdown (2026-01-01..2026-04-10) ---")
pipeline = [
    {"$match": {"symbol": sym, "period": "daily",
                "trade_date": {"$gte": "2026-01-01", "$lte": "2026-04-10"}}},
    {"$group": {"_id": {"src": "$data_source"}, "n": {"$sum": 1},
                "dmin": {"$min": "$trade_date"}, "dmax": {"$max": "$trade_date"}}},
]
for r in col.aggregate(pipeline):
    print(r)

# 3) fetch tushare+akshare rows and compare to demo parquet
demo = pd.read_parquet("data/ashare/1d/300368.SZ.parquet")
demo["date"] = pd.to_datetime(demo["datetime"]).dt.strftime("%Y-%m-%d")
demo = demo.set_index("date")
overlap_start, overlap_end = "2026-01-05", "2026-04-10"

for src in ("tushare", "akshare"):
    docs = list(col.find({"symbol": sym, "period": "daily", "data_source": src,
                          "trade_date": {"$gte": overlap_start, "$lte": overlap_end}},
                         {"_id": 0, "trade_date": 1, "open": 1, "high": 1, "low": 1, "close": 1,
                          "volume": 1, "pre_close": 1, "adj_factor": 1, "pct_chg": 1}))
    if not docs:
        print(f"\n[{src}] no rows")
        continue
    m = pd.DataFrame(docs).set_index("trade_date").sort_index()
    common = m.index.intersection(demo.index)
    if len(common) == 0:
        print(f"\n[{src}] no common dates")
        continue
    j = m.loc[common]
    d = demo.loc[common]
    def rel(a, b):
        r = (a / b).replace([float("inf"), -float("inf")], pd.NA).dropna()
        return r.min(), r.max()
    print(f"\n[{src}] common days={len(common)}")
    for f_ in ("open", "high", "low", "close"):
        lo, hi = rel(j[f_].astype(float), d[f_].astype(float))
        print(f"  mongo/demo {f_}: min={lo:.4f} max={hi:.4f}")
    lo, hi = rel(j["volume"].astype(float), d["volume"].astype(float))
    print(f"  mongo/demo volume: min={lo:.4f} max={hi:.4f}")
    print("  sample mongo row:", j.iloc[0].to_dict())
    print("  sample demo  row:", d.iloc[0][["open", "high", "low", "close", "volume"]].to_dict())

# 4) intraday_backtest_bars: bars/day per code to identify timeframe
col3 = db["intraday_backtest_bars"]
sym3 = demo_codes[0].split(".")[0]
one_day = list(col3.find({"code": sym3}, {"_id": 0}).sort("datetime", 1).limit(60))
if one_day:
    ts = pd.DatetimeIndex([x["datetime"] for x in one_day])
    diffs = ts.to_series().diff().dropna().dt.total_seconds().div(60)
    print(f"\n--- intraday_backtest_bars {sym3} ---")
    print("first 12 timestamps (UTC):", [str(t) for t in ts[:12]])
    print("bar gaps (min):", sorted(diffs.unique())[:6], "| n in first day:", (ts < ts[0].normalize() + pd.Timedelta(days=1)).sum())
