import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
import numpy as np
import pickle
import glob
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = "C:/Users/sudee/projects/Final Year Project"
PROCESSED_DIR = os.path.join(BASE_DIR, "data/processed")
MODEL_DIR = os.path.join(PROCESSED_DIR, "models")
RAW_DIR = os.path.join(BASE_DIR, "data/raw")
FEATURES_PATH = os.path.join(PROCESSED_DIR, "all_stocks_features.parquet")
CONFIG_PATH = os.path.join(PROCESSED_DIR, "fold_config.json")

with open(CONFIG_PATH) as f:
    CONFIG = json.load(f)

model_candidates = glob.glob(os.path.join(MODEL_DIR, "model_fold*.pkl"))
if not model_candidates:
    raise FileNotFoundError(f"No model files found in {MODEL_DIR}")

def fold_num(path):
    m = re.search(r"model_fold(\d+)\.pkl$", os.path.basename(path))
    return int(m.group(1)) if m else -1

MODEL_PATH = max(model_candidates, key=fold_num)

with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)
    MODEL = bundle["model"]
    SCALER = bundle["scaler"]
    FEATURE_COLS = bundle.get("feature_cols") or CONFIG.get("feature_cols")

if not FEATURE_COLS:
    raise KeyError("feature_cols missing in both model bundle and fold_config.json")

FEATURES_DF = pd.read_parquet(FEATURES_PATH)
FEATURES_DF["Date"] = pd.to_datetime(FEATURES_DF["Date"])

ALL_SYMBOLS = sorted(FEATURES_DF["Symbol"].unique().tolist())

def safe_val(v):
    if pd.isna(v) or (isinstance(v, float) and np.isinf(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return round(float(v), 4)
    return v

@app.get("/api/stocks")
def get_stocks():
    results = []

    for sym in ALL_SYMBOLS:
        stock_df = FEATURES_DF[FEATURES_DF["Symbol"] == sym]
        latest = stock_df.dropna(subset=FEATURE_COLS).sort_values("Date").tail(1)

        if latest.empty:
            continue

        row = latest.iloc[0]

        X = SCALER.transform(latest[FEATURE_COLS].values)
        conf = float(MODEL.predict_proba(X)[0, 1])

        if conf >= 0.65:
            tier = "High"
        elif conf >= 0.55:
            tier = "Medium"
        elif conf >= 0.45:
            tier = "Neutral"
        else:
            tier = "Low"

        results.append({
            "Symbol": sym,
            "Date": row["Date"].strftime("%Y-%m-%d"),
            "Close": safe_val(row["Close"]),
            "rsi": safe_val(row["RSI_14"]),
            "confidence": round(conf, 4),
            "Tier": tier
        })

    results.sort(key=lambda x: x["confidence"], reverse=True)
    return {"stocks": results, "count": len(results)}

@app.get("/api/stocks/{symbol}")
def get_stock_details(symbol: str, days: int = 180):
    symbol = symbol.upper()
    days = max(1, min(days, 2000))  # keep days in a safe range

    if symbol not in ALL_SYMBOLS:
        raise HTTPException(status_code=404, detail=f"Symbol '{symbol}' not found")

    stock_df = (FEATURES_DF[FEATURES_DF["Symbol"] == symbol]
                .sort_values("Date")
                .tail(days)
                .copy())

    if stock_df.empty:
        raise HTTPException(status_code=404, detail=f"data for symbol '{symbol}' not found")

    candles    = []
    indicators = {
        "sma20":    [],
        "bb_upper": [],
        "bb_lower": [],
        "bb_mid":   [],
        "ema12":    [],
        "ema26":    [],
        "rsi":      [],
        "macd":     [],
        "macd_sig": [],
        "macd_hist":[],
        "volume":   [],
        "dates":    [],
    }

    for _, row in stock_df.iterrows():
        date_str = str(row["Date"].date())
        indicators["dates"].append(date_str)

        # Candlestick data: date, open, high, low, close
        candles.append({
            "t": date_str,
            "o": safe_val(row["Open"]),
            "h": safe_val(row["High"]),
            "l": safe_val(row["Low"]),
            "c": safe_val(row["Close"]),
            "v": safe_val(row["Volume"]),
        })

        # Overlay indicators
        indicators["sma20"].append(safe_val(row.get("SMA_20")))
        indicators["bb_upper"].append(safe_val(row.get("BB_Upper")))
        indicators["bb_lower"].append(safe_val(row.get("BB_Lower")))
        indicators["bb_mid"].append(safe_val(row.get("BB_Middle")))
        indicators["ema12"].append(safe_val(row.get("EMA_12")))
        indicators["ema26"].append(safe_val(row.get("EMA_26")))

        # Sub-chart indicators
        indicators["rsi"].append(safe_val(row.get("RSI_14")))
        indicators["macd"].append(safe_val(row.get("MACD")))
        indicators["macd_sig"].append(safe_val(row.get("MACD_Signal")))
        macd_hist = (safe_val(row.get("MACD")) or 0) - (safe_val(row.get("MACD_Signal")) or 0)
        indicators["macd_hist"].append(round(macd_hist, 4))
        indicators["volume"].append(safe_val(row.get("Volume")))

    return {
        "symbol":     symbol,
        "days":       days,
        "candles":    candles,
        "indicators": indicators,
    }


# ════════════════════════════════════════════════════════════
# ENDPOINT 3: GET /api/signal/<symbol>
# Returns ML confidence score + full signal explanation
# for one stock. Frontend uses this for the signal panel.
# ════════════════════════════════════════════════════════════

@app.get("/api/signal/{symbol}")
def get_signal(symbol):
    symbol   = symbol.upper()

    if symbol not in ALL_SYMBOLS:
        raise HTTPException(status_code=404, detail=f"Symbol '{symbol}' not found")

    stock_df = FEATURES_DF[FEATURES_DF["Symbol"] == symbol]
    latest   = stock_df.dropna(subset=FEATURE_COLS).sort_values("Date").tail(1)

    if latest.empty:
        raise HTTPException(status_code=404, detail=f"Insufficient data for symbol '{symbol}'")

    row  = latest.iloc[0]

    # ── ML confidence ────────────────────────────────────────
    X    = SCALER.transform(latest[FEATURE_COLS].values)
    conf = float(MODEL.predict_proba(X)[0, 1])

    # ── Active signals ───────────────────────────────────────
    active_signals = []
    if safe_val(row.get("Signal_RSI_oversold")):
        active_signals.append("RSI Oversold Recovery")
    if safe_val(row.get("Signal_MACD_cross")):
        active_signals.append("MACD Bullish Crossover")
    if safe_val(row.get("Signal_BB_lower")):
        active_signals.append("BB Lower Band Touch")

    # ── Indicator context ────────────────────────────────────
    # These are the human-readable values shown in the signal panel
    rsi_val  = safe_val(row.get("RSI_14"))
    rsi_zone = ("oversold"   if rsi_val and rsi_val < 30 else
                "overbought" if rsi_val and rsi_val > 70 else
                "neutral")

    macd_val  = safe_val(row.get("MACD"))
    macd_sig  = safe_val(row.get("MACD_Signal"))
    macd_hist = round((macd_val or 0) - (macd_sig or 0), 4)
    macd_bias = "bullish" if macd_hist > 0 else "bearish"

    bb_pctb   = safe_val(row.get("BB_pctB"))
    bb_zone   = ("below lower band" if bb_pctb and bb_pctb < 0 else
                 "above upper band" if bb_pctb and bb_pctb > 1 else
                 "within bands")

    in_uptrend = bool(safe_val(row.get("In_uptrend")))
    vol_ratio  = safe_val(row.get("Volume_ratio"))
    vol_note   = ("high volume" if vol_ratio and vol_ratio > 2 else
                  "low volume"  if vol_ratio and vol_ratio < 0.5 else
                  "normal volume")

    # ── Confidence interpretation ────────────────────────────
    if conf >= 0.65:
        verdict     = "Strong buy signal"
        description = ("Model is highly confident this signal has a good "
                       "probability of producing a positive 10-day return "
                       "after transaction costs.")
        color       = "green"
    elif conf >= 0.55:
        verdict     = "Moderate signal"
        description = ("Model sees a moderate edge. Consider position sizing "
                       "conservatively and confirming with volume.")
        color       = "orange"
    elif conf >= 0.45:
        verdict     = "Neutral — no clear edge"
        description = ("Model cannot identify a reliable edge at this time. "
                       "Signal does not clear the recommended threshold.")
        color       = "gray"
    else:
        verdict     = "Weak — signal likely to fail"
        description = ("Model assigns low probability to this signal succeeding. "
                       "Historical patterns suggest avoiding this trade.")
        color       = "red"

    return {
        "symbol":          symbol,
        "date":            str(row["Date"].date()),
        "close":           safe_val(row["Close"]),
        "confidence":      round(conf, 3),
        "verdict":         verdict,
        "verdict_color":   color,
        "description":     description,
        "active_signals":  active_signals,
        "indicators": {
            "rsi":         rsi_val,
            "rsi_zone":    rsi_zone,
            "macd":        macd_val,
            "macd_signal": macd_sig,
            "macd_hist":   macd_hist,
            "macd_bias":   macd_bias,
            "bb_pctb":     bb_pctb,
            "bb_zone":     bb_zone,
            "in_uptrend":  in_uptrend,
            "volume_ratio":vol_ratio,
            "volume_note": vol_note,
        },
        "thresholds": {
            "recommended": 0.60,
            "minimum":     0.55,
        }
    }


# ════════════════════════════════════════════════════════════
# ENDPOINT 4: GET /api/summary
# Returns the top 10 high-confidence stocks right now.
# Frontend uses this for the dashboard overview panel.
# ════════════════════════════════════════════════════════════

@app.get("/api/summary")
def get_summary():
    results = []

    for symbol in ALL_SYMBOLS:
        stock_df = FEATURES_DF[FEATURES_DF["Symbol"] == symbol]
        latest   = stock_df.dropna(subset=FEATURE_COLS).sort_values("Date").tail(1)
        if latest.empty:
            continue

        row  = latest.iloc[0]
        X    = SCALER.transform(latest[FEATURE_COLS].values)
        conf = float(MODEL.predict_proba(X)[0, 1])

        if conf < 0.55:
            continue   # Only include stocks above minimum threshold

        active = []
        if safe_val(row.get("Signal_RSI_oversold")): active.append("RSI")
        if safe_val(row.get("Signal_MACD_cross")):   active.append("MACD")
        if safe_val(row.get("Signal_BB_lower")):     active.append("BB")

        results.append({
            "symbol":     symbol,
            "close":      safe_val(row["Close"]),
            "date":       str(row["Date"].date()),
            "rsi":        safe_val(row.get("RSI_14")),
            "confidence": round(conf, 3),
            "signals":    active,
            "in_uptrend": bool(safe_val(row.get("In_uptrend"))),
        })

    results.sort(key=lambda x: x["confidence"], reverse=True)
    return {
        "top_signals": results[:10],
        "total_above_threshold": len(results),
        "threshold_used": 0.55,
    }
