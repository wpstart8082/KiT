# -*- coding: utf-8 -*-
"""Check: adj_factor presence, minute-bar conventions, backtest-bar timeframe, code overlap."""
import json
import os

import pandas as pd
from pymongo import MongoClient

URI = os.environ.get("MONGO_URI", "mongodb://127.0.0.1:27017/tradingagentscn")
db = MongoClient(URI, serverSelectionTimeoutMS=8000)["tradingagentscn"]
instruments = json.loads(open("data/instruments.json", encoding="utf-8").read())
demo6 = [c.split(".")[0] for c in instruments]

# 1) adj_factor present anywhere?
print("docs with non-null adj_factor:", db.stock_daily_quotes.count_documents(
    {"period": "daily", "adj_factor": {"$ne": None}}, limit=... ) if False else
    db.stock_daily_quotes.count_documents({"period": "daily", "adj_factor": {"$ne": None}}))

# 2) demo 5m parquet timestamp convention (one day)
d5 = pd.read_parquet("data/ashare/5m/300368.SZ.parquet")
d5["dt"] = pd.to_datetime(d5["datetime"])
day = d5[d5["dt"].dt.date == pd.Timestamp("2026-03-04").date()]
print("\ndemo 5m bars on 2026-03-04:", len(day))
print("first 4:", [str(t) for t in day["dt"].head(4)])
print("last 4:", [str(t) for t in day["dt"].tail(4)])

# also demo 15m / 30m / 1h / 2h one day
for sc in ("15m", "30m", "1h", "2h"):
    dd = pd.read_parquet(f"data/ashare/{sc}/300368.SZ.parquet")
    dd["dt"] = pd.to_datetime(dd["datetime"])
    day = dd[dd["dt"].dt.date == pd.Timestamp("2026-03-04").date()]
    print(f"demo {sc} on 2026-03-04: n={len(day)} ts={[t.strftime('%H:%M') for t in day['dt']]}")

# 3) minute collections: overlap with demo 100
mb_codes = set(db.stock_minute_bars.distinct("code"))
bb_codes = set(db.intraday_backtest_bars.distinct("code"))
print("\nstock_minute_bars codes:", len(mb_codes), "| demo overlap:", len(mb_codes & set(demo6)))
print("intraday_backtest_bars codes:", len(bb_codes), "| demo overlap:", len(bb_codes & set(demo6)))
ov = sorted(bb_codes & set(demo6))
print("backtest overlap sample:", ov[:10])

# 4) backtest bars timeframe: pick an overlapping code, one day
if ov:
    code = ov[0]
    docs = list(db.intraday_backtest_bars.find({"code": code}, {"_id": 0}).sort("datetime", 1))
    ts = pd.DatetimeIndex([x["datetime"] for x in docs])
    d0 = ts[0].normalize()
    day1 = ts[(ts >= d0) & (ts < d0 + pd.Timedelta(days=1))]
    print(f"\nbacktest bars {code} first day: n={len(day1)} ts={[t.strftime('%H:%M') for t in day1]}")
    diffs = pd.Series(day1).diff().dropna().dt.total_seconds().div(60).unique()
    print("intraday gaps (min):", sorted(diffs))

# 5) stats.json structure (feature names -> scale-free?)
st = json.loads(open("data/stats.json", encoding="utf-8").read())
print("\nstats.json top keys:", list(st)[:10])
import itertools
k0 = list(st)[0]
print("stats[%s]:" % k0, json.dumps(st[k0], ensure_ascii=False)[:400])
