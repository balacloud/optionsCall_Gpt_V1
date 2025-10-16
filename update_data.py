x#!/usr/bin/env python3
import os
import sys
import time
from datetime import datetime, timedelta, timezone
import pandas as pd
import yfinance as yf
import numpy as np
import json

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
TICKERS_FILE = os.path.join(REPO_ROOT, "tickers.txt")

LOOKBACK_DAYS = int(os.environ.get("LOOKBACK_DAYS", "730"))
MAX_RETRIES = 3
RETRY_SLEEP = 3

def read_tickers(path):
    if not os.path.exists(path):
        sys.exit("tickers.txt not found")
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]

def fetch_one(ticker):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=LOOKBACK_DAYS)
    for i in range(MAX_RETRIES):
        try:
            df = yf.download(ticker, start=start.date().isoformat(),
                             end=end.date().isoformat(),
                             interval="1d", progress=False, threads=False)
            if df is None or df.empty:
                raise ValueError("Empty result")
            df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
            df = df.reset_index()
            if "Date" in df.columns:
                df["date"] = df["Date"].dt.date
            else:
                df["date"] = df.index.date
            df = df[["date","open","high","low","close","volume"]].dropna()
            df = df.drop_duplicates(subset=["date"]).sort_values("date")
            return df
        except Exception as e:
            print(f"Warn {ticker} attempt {i}: {e}")
            time.sleep(RETRY_SLEEP)
    return None

def compute_sr_levels(df, lookback=8, cluster_pct=0.6, max_levels=6):
    sup, res = [], []
    lows = df["low"].to_numpy()
    highs = df["high"].to_numpy()
    n = len(df)
    for i in range(lookback, n - lookback):
        window_low = lows[i-lookback:i+lookback+1]
        window_high = highs[i-lookback:i+lookback+1]
        if df["low"].iloc[i] == np.min(window_low):
            sup.append(df["low"].iloc[i])
        if df["high"].iloc[i] == np.max(window_high):
            res.append(df["high"].iloc[i])
    def cluster(levels):
        levels = sorted(levels)
        out = []
        for p in levels:
            if not out or abs(p - out[-1])/p > (cluster_pct/100):
                out.append(p)
            else:
                out[-1] = (out[-1] + p)/2
        return out[-max_levels:]
    return {"support": cluster(sup), "resistance": cluster(res)}

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100/(1 + rs))

def compute_metrics(df):
    closes = df["close"].astype(float)
    df["dma20"] = closes.rolling(20).mean()
    df["dma50"] = closes.rolling(50).mean()
    df["dma200"] = closes.rolling(200).mean()
    df["rsi14"] = compute_rsi(closes, 14)
    latest = df.iloc[-1]
    bias = "Neutral"
    if latest["close"] > latest["dma20"] > latest["dma50"] > latest["dma200"]:
        bias = "Bullish"
    elif latest["close"] < latest["dma20"] < latest["dma50"] < latest["dma200"]:
        bias = "Bearish"
    momentum = "Rising" if (df["dma20"].iloc[-1] - df["dma20"].iloc[-5]) > 0 else "Falling"
    return {
        "last_date": str(latest["date"]),
        "close": float(latest["close"]),
        "dma20": float(latest["dma20"]),
        "dma50": float(latest["dma50"]),
        "dma200": float(latest["dma200"]),
        "rsi14": float(latest["rsi14"]),
        "momentum": momentum,
        "trend_bias": bias
    }

def save_sr_json(ticker, df):
    sr = compute_sr_levels(df)
    with open(os.path.join(DATA_DIR, f"SR_{ticker}.json"), "w") as f:
        json.dump(sr, f, indent=2)

def save_metrics_json(ticker, df):
    m = compute_metrics(df)
    with open(os.path.join(DATA_DIR, f"metrics_{ticker}.json"), "w") as f:
        json.dump(m, f, indent=2)

def save_summary_json(ticker, df):
    sr_path = os.path.join(DATA_DIR, f"SR_{ticker}.json")
    metrics_path = os.path.join(DATA_DIR, f"metrics_{ticker}.json")
    sr = {}
    metrics = {}
    if os.path.exists(sr_path):
        with open(sr_path) as f:
            sr = json.load(f)
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
    last = df.iloc[-1].to_dict()
    summary = {
        "ticker": ticker,
        "last_date": str(last["date"]),
        "close": float(last["close"]),
        "open": float(last["open"]),
        "high": float(last["high"]),
        "low": float(last["low"]),
        "volume": int(last["volume"]),
        **metrics,
        **sr
    }
    with open(os.path.join(DATA_DIR, f"summary_{ticker}.json"), "w") as f:
        json.dump(summary, f, indent=2)

def merge_with_existing(ticker, newdf):
    path = os.path.join(DATA_DIR, f"{ticker}.csv")
    if os.path.exists(path):
        old = pd.read_csv(path)
        old["date"] = pd.to_datetime(old["date"]).dt.date
        merged = pd.concat([old, newdf], ignore_index=True)
        merged = merged.drop_duplicates(subset=["date"], keep="last").sort_values("date")
    else:
        merged = newdf
    os.makedirs(DATA_DIR, exist_ok=True)
    merged.to_csv(path, index=False)
    save_sr_json(ticker, merged)
    save_metrics_json(ticker, merged)
    save_summary_json(ticker, merged)

def main():
    tickers = read_tickers(TICKERS_FILE)
    for t in tickers:
        df = fetch_one(t)
        if df is None or df.empty:
            continue
        merge_with_existing(t, df)

if __name__ == "__main__":
    main()
