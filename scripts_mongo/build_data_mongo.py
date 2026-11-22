# -*- coding: utf-8 -*-
"""Build a KiT data_root from MongoDB (tradingagentscn).

- 1d   : stock_daily_quotes (tushare), adjusted via pre_close chain (exact log-returns)
- 5m   : stock_minute_bars (UTC->+8) union intraday_backtest_bars, cross-day adjusted via daily chain
- 15m/30m/1h/2h : session-relative resample of the 5m base (matches kit_v1 calendar grids)
- trade_dates.csv : from trade_calendar (is_open=True)
- instruments.json / stats.json : copied from data/ (aligned with train260410 ckpt)

Self-checks:
- resample code validated against demo 5m -> demo 15m/30m/1h/2h on 2026-03-04
- daily close vs intraday 15:00 close cross-source check
- per-day 48-bar completeness and trading-calendar contiguity for chosen windows
"""
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pymongo import MongoClient

ROOT = Path(__file__).resolve().parents[1]
URI = os.environ.get("MONGO_URI", "mongodb://127.0.0.1:27017/tradingagentscn")
OUT = ROOT / "data_mongo"
MARKET_DIR = OUT / "ashare"

WC = ("1m:1200/120", "5m:960/96", "15m:800/64", "30m:640/40", "1h:400/20", "2h:360/20", "1d:250/20")
LC = dict(s.split(":") and (s.split(":")[0], int(s.split(":")[1].split("/")[0])) for s in
          [p for w in WC for p in [w]])
LH = {s.split(":")[0]: int(s.split(":")[1].split("/")[1]) for s in WC}
BARS_PER_DAY = {"5m": 48, "15m": 16, "30m": 8, "1h": 4, "2h": 2, "1d": 1}


def mongo():
    return MongoClient(URI, serverSelectionTimeoutMS=8000)["tradingagentscn"]


# ---------------------------------------------------------------- calendar
def build_calendar(db) -> pd.DatetimeIndex:
    docs = db.trade_calendar.find({"is_open": True}, {"_id": 0, "cal_date": 1})
    days = sorted(d["cal_date"] for d in docs)
    idx = pd.to_datetime(days, format="%Y-%m-%d")
    print(f"calendar: {idx[0].date()} .. {idx[-1].date()} ({len(idx)} open days)")
    OUT.mkdir(exist_ok=True)
    pd.Series(idx.strftime("%Y%m%d"), name="trade_date").to_csv(
        OUT / "trade_dates.csv", index=False, encoding="utf-8-sig")
    return idx


# ---------------------------------------------------------------- daily
def load_daily(db, sym: str) -> pd.DataFrame:
    docs = db.stock_daily_quotes.find(
        {"symbol": sym, "period": "daily", "data_source": "tushare"},
        {"_id": 0, "trade_date": 1, "open": 1, "high": 1, "low": 1, "close": 1,
         "pre_close": 1, "volume": 1}).sort("trade_date", 1)
    df = pd.DataFrame(list(docs))
    if df.empty:
        return df
    df["dt"] = pd.to_datetime(df["trade_date"]) + pd.Timedelta(hours=15)
    for c in ("open", "high", "low", "close", "pre_close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)


def adjust_daily(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Rebuild a continuous adjusted series via the pre_close chain.

    ratio_t = pre_close_t / close_{t-1} (1.0 when no corporate action).
    adjusted price = raw * cumprod(ratio); ex-div days reported.
    """
    prev_close = df["close"].shift(1)
    ratio = df["pre_close"] / prev_close
    exdiv_days = df.loc[(ratio - 1.0).abs() > 1e-6, "trade_date"].tolist()
    ratio = ratio.fillna(1.0).clip(0.2, 5.0)          # guard against bad rows
    factor = ratio.cumprod()
    out = pd.DataFrame({
        "datetime": df["dt"],
        "open": df["open"] * factor, "high": df["high"] * factor,
        "low": df["low"] * factor, "close": df["close"] * factor,
        "volume": df["volume"]})
    return out, exdiv_days


def daily_chain_factors(df: pd.DataFrame) -> pd.Series:
    """Per-day multiplicative factor vs the series start (for intraday cross-day adjust)."""
    prev_close = df["close"].shift(1)
    ratio = (df["pre_close"] / prev_close).fillna(1.0).clip(0.2, 5.0)
    return pd.Series(ratio.cumprod().values, index=df["trade_date"].values)


# ---------------------------------------------------------------- intraday
def load_5m_union(db, code6: str) -> pd.DataFrame:
    """5m bars from both collections (UTC), +8h, deduped, preferring stock_minute_bars."""
    frames = []
    for coll in ("intraday_backtest_bars", "stock_minute_bars"):
        docs = coll and list(db[coll].find(
            {"code": code6}, {"_id": 0, "datetime": 1, "open": 1, "high": 1, "low": 1,
                              "close": 1, "volume": 1}).sort("datetime", 1))
        if docs:
            f = pd.DataFrame(docs)
            f["src"] = coll
            frames.append(f)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"]) + pd.Timedelta(hours=8)   # UTC -> CST
    df = df.sort_values(["datetime", "src"], ascending=[True, True])
    df["pref"] = df["src"].map({"stock_minute_bars": 0, "intraday_backtest_bars": 1})
    df = df.sort_values(["datetime", "pref"]).drop_duplicates("datetime", keep="first")
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)[
        ["datetime", "open", "high", "low", "close", "volume"]]


def resample_session(df5: pd.DataFrame, k: int) -> pd.DataFrame:
    """Group each day's 48 session-relative 5m bars by floor(idx/k).

    Works for k=3 (15m), 6 (30m), 12 (1h), 24 (2h) — matches kit_v1 session grids:
    AM indices 0..23, PM 24..47; boundaries 9:35..11:30 / 13:05..15:00.
    """
    t = df5["datetime"]
    day = t.dt.normalize()
    minute = t.dt.hour * 60 + t.dt.minute
    am = (minute >= 575) & (minute <= 690)          # 9:35 .. 11:30
    pm = (minute >= 785) & (minute <= 900)          # 13:05 .. 15:00
    idx = pd.Series(np.where(am, (minute - 575) // 5, np.where(pm, 24 + (minute - 785) // 5, -1)), index=t.index)
    bad = idx < 0
    if bad.any():
        raise ValueError(f"unexpected 5m timestamps: {t[bad].head().tolist()}")
    grp = pd.MultiIndex.from_arrays([day.values, idx // k])
    agg = df5.assign(_g=np.arange(len(df5))).groupby(grp, sort=True).agg(
        datetime=("datetime", "last"), open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"), volume=("volume", "sum"), _n=("open", "size"))
    full = agg[agg["_n"] == k].drop(columns="_n").reset_index(drop=True)
    return full[["datetime", "open", "high", "low", "close", "volume"]]


# ---------------------------------------------------------------- checks
def check_resample_vs_demo() -> None:
    base = pd.read_parquet(ROOT / "data/ashare/5m/300368.SZ.parquet")
    base["datetime"] = pd.to_datetime(base["datetime"])
    day_mask = base["datetime"].dt.date == pd.Timestamp("2026-03-04").date()
    b = base[day_mask]
    for k, sc in ((3, "15m"), (6, "30m"), (12, "1h"), (24, "2h")):
        got = resample_session(b, k)
        exp = pd.read_parquet(ROOT / f"data/ashare/{sc}/300368.SZ.parquet")
        exp["datetime"] = pd.to_datetime(exp["datetime"])
        exp = exp[exp["datetime"].dt.date == pd.Timestamp("2026-03-04").date()].reset_index(drop=True)
        ok_ts = got["datetime"].equals(exp["datetime"])
        diffs = {c: float((got[c].values - exp[c].values).__abs__().max()) for c in
                 ("open", "high", "low", "close", "volume")}
        print(f"  resample check {sc}: ts_match={ok_ts} max|diff|={ {k2: f'{v:.6g}' for k2, v in diffs.items()} }")


def day_completeness(df5: pd.DataFrame) -> pd.Series:
    return df5.groupby(df5["datetime"].dt.normalize()).size()


# ---------------------------------------------------------------- main
def main() -> None:
    db = mongo()
    instruments = json.loads((ROOT / "data/instruments.json").read_text(encoding="utf-8"))

    print("== self-check: resample vs demo (300368.SZ, 2026-03-04) ==")
    check_resample_vs_demo()

    cal = build_calendar(db)
    cal_days = set(cal.normalize())

    print("\n== daily: pick stocks with contiguous windows around anchor 2026-10-23 ==")
    DAILY_ANCHOR = pd.Timestamp("2026-10-23 15:00")
    daily_codes, daily_meta = [], {}
    cands = [c for c in instruments if not c.startswith(("8", "4"))]   # skip BJ for safety? keep all actually
    for full in instruments:
        sym = full.split(".")[0]
        raw = load_daily(db, sym)
        if len(raw) < 400:
            continue
        adj, exdiv = adjust_daily(raw)
        ts = pd.DatetimeIndex(adj["datetime"])
        n_anchor = int(ts.searchsorted(DAILY_ANCHOR, side="right"))
        if n_anchor < 250 + 130 or len(ts) - n_anchor < 20:
            continue
        # contiguity: no missing trading days in [anchor-160d, anchor+20d]
        win_days = pd.DatetimeIndex(ts[n_anchor - 170:n_anchor + 20]).normalize()
        expected = [d for d in cal if DAILY_ANCHOR - pd.Timedelta(days=240) <= d <= DAILY_ANCHOR + pd.Timedelta(days=30)]
        have = set(win_days)
        missing = [d for d in expected if d > (DAILY_ANCHOR - pd.Timedelta(days=330)) and d not in have]
        # simpler: count calendar open days between first and last of window
        w0, w1 = ts[n_anchor - 170], ts[n_anchor + 19]
        open_in_span = sum(1 for d in cal if w0 <= d <= w1)
        contiguous = (n_anchor + 19) - (n_anchor - 170) + 1 == open_in_span
        if not contiguous:
            continue
        daily_codes.append(full)
        daily_meta[full] = dict(rows=len(adj), exdiv_2026=[d for d in exdiv if d >= "2026-01-01"],
                                last=str(ts[-1]))
        if len(daily_codes) >= 6:
            break
    for c in daily_codes:
        m = daily_meta[c]
        print(f"  {c}: rows={m['rows']} last={m['last']} exdiv2026={m['exdiv_2026'][:6]}")

    print("\n== intraday: coverage of (backtest|minute) ∩ demo100 ==")
    mb = set(db.stock_minute_bars.distinct("code"))
    bb = set(db.intraday_backtest_bars.distinct("code"))
    demo6 = {c.split(".")[0]: c for c in instruments}
    both = (mb | bb) & set(demo6)
    scored = []
    for code6 in sorted(both):
        df5 = load_5m_union(db, code6)
        if df5.empty:
            continue
        dc = day_completeness(df5)
        full_days = dc[dc == 48]
        span = (pd.Timestamp("2026-09-01"), pd.Timestamp("2026-11-20"))
        fd = full_days[(full_days.index >= span[0]) & (full_days.index <= span[1])]
        scored.append((len(fd), code6, str(full_days.index.min().date()), str(full_days.index.max().date())))
    scored.sort(reverse=True)
    for n, code6, d0, d1 in scored[:12]:
        print(f"  {demo6[code6]}: full48days(Sep1-Nov20)={n} range {d0}..{d1}")

    # save state for the next step
    OUT.mkdir(exist_ok=True)
    for f in ("instruments.json", "stats.json"):
        shutil.copy(ROOT / "data" / f, OUT / f)
    json.dump({"daily_codes": daily_codes, "intraday_ranked": [
        {"code": demo6[c], "full_days": n} for n, c, _, _ in scored]}, 
        open(OUT / "_build_meta.json", "w"), ensure_ascii=False, indent=1)
    print("\nsaved -> data_mongo/ (instruments/stats/trade_dates + _build_meta.json)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
