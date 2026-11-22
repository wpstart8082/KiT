# -*- coding: utf-8 -*-
"""Build final parquets for the KiT test from MongoDB and pick feasible anchors.

Outputs under data_mongo/:
  trade_dates.csv          demo calendar ∪ MongoDB trade_calendar
  instruments.json/stats.json (copied, train260410-aligned)
  ashare/1d/<code>.parquet    adjusted via pre_close chain, full history
  ashare/{5m,15m,30m,1h}/<code>.parquet  for top intraday codes
  _anchors.json           chosen cases for eval/infer_for_pred.py
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pymongo import MongoClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts_mongo"))
from build_data_mongo import (adjust_daily, daily_chain_factors, load_5m_union,
                              load_daily, mongo, resample_session)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data_mongo"
MARKET_DIR = OUT / "ashare"

WIN = {"1m": (1200, 120), "5m": (960, 96), "15m": (800, 64), "30m": (640, 40),
       "1h": (400, 20), "2h": (360, 20), "1d": (250, 20)}
BPD = {"1m": 240, "5m": 48, "15m": 16, "30m": 8, "1h": 4, "2h": 2, "1d": 1}
HL_DAYS = 5.0


def keep_bars(scale: str) -> int:
    lc, _ = WIN[scale]
    return lc + int(np.ceil(12.0 * HL_DAYS * BPD[scale])) + 8


def union_calendar(db) -> pd.DatetimeIndex:
    old = pd.read_csv(ROOT / "data/trade_dates.csv", encoding="utf-8-sig", dtype=str).iloc[:, 0]
    old_days = pd.to_datetime(old, format="%Y%m%d")
    new_days = pd.to_datetime([d["cal_date"] for d in db.trade_calendar.find({"is_open": True})])
    idx = pd.DatetimeIndex(sorted(set(old_days) | set(new_days)))
    print(f"calendar union: {idx[0].date()} .. {idx[-1].date()} ({len(idx)} days)")
    pd.Series(idx.strftime("%Y%m%d"), name="trade_date").to_csv(
        OUT / "trade_dates.csv", index=False, encoding="utf-8-sig")
    return idx


def pick_daily_stocks(db, instruments, cal):
    """Relaxed: need Lc+keep bars before anchor and a gap-free future window."""
    anchor = pd.Timestamp("2026-10-23 15:00")
    cal_set = set(cal.normalize())
    chosen = []
    for full in instruments:
        sym = full.split(".")[0]
        raw = load_daily(db, sym)
        if len(raw) < 400:
            continue
        adj, exdiv = adjust_daily(raw)
        ts = pd.DatetimeIndex(adj["datetime"])
        n = int(ts.searchsorted(anchor, side="right"))
        if n < keep_bars("1d"):
            continue
        fut_days = sorted(d for d in cal_set if anchor.normalize() < d <= ts[-1].normalize())
        if len(fut_days) < 20:
            continue
        # gap-free future: the 20 trading days after anchor must equal actual rows
        fut_rows = pd.DatetimeIndex(ts[n:n + 20]).normalize()
        if list(fut_rows) != fut_days[:20]:
            continue
        chosen.append((full, adj, [d for d in exdiv if d >= "2026-06-01"]))
        if len(chosen) == 6:
            break
    return chosen


def build_intraday(db, code_full: str, daily_raw: pd.DataFrame, chain: pd.Series):
    code6 = code_full.split(".")[0]
    df5 = load_5m_union(db, code6)
    if df5.empty:
        return {}
    # cross-day adjustment via daily pre_close chain (ratios, scale-free)
    day_key = df5["datetime"].dt.strftime("%Y-%m-%d")
    base_f = chain.get(day_key.iloc[0], 1.0)
    f = day_key.map(chain).astype(float) / float(base_f)
    f = f.fillna(1.0)
    for c in ("open", "high", "low", "close"):
        df5[c] = df5[c] * f

    out = {"5m": df5}
    for k, sc in ((3, "15m"), (6, "30m"), (12, "1h")):
        out[sc] = resample_session(df5, k)
    return out


def full_day_map(df) -> pd.Series:
    """Trading days with the exact per-day bar count."""
    return df.groupby(df["datetime"].dt.normalize()).size()


def pick_intraday_anchor(df, scale, cal_set, last_ok_day):
    """Latest anchor whose future Lh bars match the calendar projection on full days."""
    lc, lh = WIN[scale]
    per_day = BPD[scale]
    dc = full_day_map(df)
    full = dc[dc == per_day]
    ts = pd.DatetimeIndex(df["datetime"])
    n_total = len(ts)
    if n_total < lc + lh:
        return None
    for end_day in sorted(full.index[::-1]):
        n = int(ts.searchsorted(end_day + pd.Timedelta(hours=15, minutes=1), side="right"))
        if n < lc or n_total - n < lh:
            continue
        # future = next lh bars; require they form complete days per calendar
        fut_ts = ts[n:n + lh]
        fd = pd.DatetimeIndex(fut_ts).normalize()
        days_seq = []
        for d, grp in pd.Series(1, index=fd).groupby(fd):
            days_seq.append((d, len(grp)))
        ok = all(cnt == per_day for _, cnt in days_seq)
        # each future day must be an open trading day and consecutive in calendar
        cal_after = sorted(d for d in cal_set if d > end_day.normalize())
        ok = ok and [d for d, _ in days_seq] == cal_after[:len(days_seq)]
        if ok and len(days_seq) == int(np.ceil(lh / per_day)):
            return pd.Timestamp(fut_ts[0 - lh]) if False else pd.Timestamp(ts[n - 1]), n, days_seq
    return None


def main() -> None:
    db = mongo()
    instruments = json.loads((ROOT / "data/instruments.json").read_text(encoding="utf-8"))
    cal = union_calendar(db)
    cal_set = set(cal.normalize())

    print("\n== daily stocks ==")
    MARKET_DIR.mkdir(parents=True, exist_ok=True)
    daily = pick_daily_stocks(db, instruments, cal)
    anchors = []
    for full, adj, exdiv in daily:
        p = MARKET_DIR / "1d" / f"{full}.parquet"
        p.parent.mkdir(exist_ok=True)
        adj.to_parquet(p, index=False)
        print(f"  {full}: {len(adj)} rows -> {p.name} | exdiv since Jun26: {exdiv}")
        anchors.append(dict(case=f"{full},1d,2026-10-23"))

    print("\n== intraday stocks (top coverage) ==")
    intr_codes = ["603686.SH", "601088.SH", "002415.SZ"]
    for full in intr_codes:
        code6 = full.split(".")[0]
        raw = load_daily(db, code6)
        if raw.empty:
            print(f"  {full}: no daily rows, skip")
            continue
        chain = daily_chain_factors(raw)
        frames = build_intraday(db, full, raw, chain)
        if not frames:
            print(f"  {full}: no 5m data")
            continue
        # cross-source sanity: 15:00 5m close vs tushare daily close (raw both)
        d5raw = load_5m_union(db, code6)
        eod = d5raw[d5raw["datetime"].dt.time == pd.Timestamp("15:00").time()]
        cmp_ = eod.set_index(eod["datetime"].dt.strftime("%Y-%m-%d"))["close"]
        dr = raw.set_index("trade_date")["close"]
        common = cmp_.index.intersection(dr.index)
        if len(common):
            rel = (cmp_[common].astype(float) / dr[common].astype(float))
            print(f"  {full}: 5mEOD/daily close ratio min={rel.min():.4f} max={rel.max():.4f} n={len(common)}")
        total5 = len(frames["5m"])
        print(f"  {full}: 5m bars={total5} | " + " ".join(
            f"{sc}={len(v)}" for sc, v in frames.items()))
        for sc in ("5m", "15m", "30m", "1h"):
            df = frames[sc]
            p = MARKET_DIR / sc / f"{full}.parquet"
            p.parent.mkdir(exist_ok=True)
            df.to_parquet(p, index=False)
            r = pick_intraday_anchor(df, sc, cal_set, None)
            if r:
                anchor_ts, n, days_seq = r
                tag = anchor_ts.strftime("%Y-%m-%d %H:%M")
                print(f"    {sc}: anchor={tag} fut_days={[str(d.date()) for d, _ in days_seq]}")
                anchors.append(dict(case=f'{full},{sc},"{tag}"'))
            else:
                print(f"    {sc}: no feasible anchor (need Lc={WIN[sc][0]}+Lh={WIN[sc][1]})")
        exd = [d for d in daily_chain_factors(raw).index]  # placeholder no-op
        _, exdiv = adjust_daily(raw)
        print(f"    exdiv since Jun26: {[d for d in exdiv if d >= '2026-06-01']}")

    for f in ("instruments.json", "stats.json"):
        shutil.copy(ROOT / "data" / f, OUT / f)
    json.dump(anchors, open(OUT / "_anchors.json", "w"), ensure_ascii=False, indent=1)
    print(f"\nanchors chosen: {len(anchors)} -> data_mongo/_anchors.json")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
