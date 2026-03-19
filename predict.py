"""
Stock Price Prediction — LSTM with TensorFlow
=============================================
Predicts next 365 days for AAPL, GOOGL, MSFT, META
using a 2-layer LSTM model trained on historical data.

Free data sources only:
  - yfinance  (Yahoo Finance)
  - FRED API  (macro indicators, free key at fred.stlouisfed.org)

Run:
    python predict.py
    python predict.py --tickers AAPL TSLA NVDA
    python predict.py --tickers AAPL --days 180
"""

import argparse
import warnings
import os
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import yfinance as yf

warnings.filterwarnings("ignore")
os.makedirs("plots", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# ──────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────
DEFAULT_TICKERS = ["AAPL", "GOOGL", "MSFT", "META"]
WINDOW_SIZE     = 60      # days of history fed into each prediction
FORECAST_DAYS   = 365     # prediction horizon
EPOCHS          = 100
BATCH_SIZE      = 32
TRAIN_SPLIT     = 0.80    # 80% train / 20% test
LOOKBACK_YEARS  = 5       # years of historical data to download

COLORS = {
    "AAPL":  "#1565C0",
    "GOOGL": "#2E7D32",
    "MSFT":  "#6A1B9A",
    "META":  "#E65100",
}

# ──────────────────────────────────────────────
#  DATA
# ──────────────────────────────────────────────

def fetch_stock_data(ticker: str, years: int = LOOKBACK_YEARS) -> pd.DataFrame:
    """Download OHLCV from Yahoo Finance and engineer features."""
    end   = datetime.today()
    start = end - timedelta(days=years * 365)
    print(f"  Downloading {ticker} from Yahoo Finance ({start.date()} → {end.date()})...")

    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)

    # ── Technical indicators (all computed from free OHLCV data) ──
    df["SMA_20"]  = df["Close"].rolling(20).mean()
    df["SMA_50"]  = df["Close"].rolling(50).mean()
    df["EMA_12"]  = df["Close"].ewm(span=12).mean()
    df["EMA_26"]  = df["Close"].ewm(span=26).mean()
    df["MACD"]    = df["EMA_12"] - df["EMA_26"]
    df["Signal"]  = df["MACD"].ewm(span=9).mean()

    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    bb_mid      = df["Close"].squeeze().rolling(20).mean()
    bb_std      = df["Close"].squeeze().rolling(20).std()
    df["BB_upper"] = (bb_mid + 2 * bb_std).values
    df["BB_lower"] = (bb_mid - 2 * bb_std).values
    df["BB_width"] = ((bb_mid + 2 * bb_std - (bb_mid - 2 * bb_std)) / bb_mid).values

    df["Volume_change"] = df["Volume"].pct_change()
    df["Daily_return"]  = df["Close"].pct_change()
    df["Volatility"]    = df["Daily_return"].rolling(20).std()

    df.dropna(inplace=True)
    df.to_csv(f"data/{ticker}_raw.csv")
    print(f"  Saved {len(df)} rows → data/{ticker}_raw.csv")
    return df


# ──────────────────────────────────────────────
#  PREPROCESSING
# ──────────────────────────────────────────────

FEATURE_COLS = [
    "Close", "Volume", "SMA_20", "SMA_50", "MACD",
    "Signal", "RSI", "BB_width", "Volatility", "Daily_return"
]

def build_sequences(data: np.ndarray, window: int):
    """Slide a window across data to create (X, y) pairs."""
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window:i])
        y.append(data[i, 0])          # predict Close (column 0)
    return np.array(X), np.array(y)


def prepare_data(df: pd.DataFrame, window: int = WINDOW_SIZE, train_split: float = TRAIN_SPLIT):
    features = df[FEATURE_COLS].values
    scaler   = MinMaxScaler()
    scaled   = scaler.fit_transform(features)

    X, y = build_sequences(scaled, window)
    split = int(len(X) * train_split)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    return X_train, X_test, y_train, y_test, scaler, scaled


# ──────────────────────────────────────────────
#  MODEL
# ──────────────────────────────────────────────

def build_model(window: int, n_features: int) -> tf.keras.Model:
    """
    2-layer Bidirectional LSTM with dropout regularisation.

    Architecture:
        Input  (window, n_features)
        BiLSTM (128 units, return_sequences=True)
        Dropout 0.2
        BiLSTM (64 units)
        Dropout 0.2
        Dense  (32, relu)
        Dense  (1)  ← predicted Close (normalised)
    """
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True), input_shape=(window, n_features)),
        Dropout(0.2),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="huber")
    return model


def train_model(model, X_train, y_train, X_test, y_test):
    callbacks = [
        EarlyStopping(patience=12, restore_best_weights=True, monitor="val_loss"),
        ReduceLROnPlateau(factor=0.5, patience=6, min_lr=1e-6, monitor="val_loss"),
    ]
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=0,
    )
    return history


# ──────────────────────────────────────────────
#  FORECASTING
# ──────────────────────────────────────────────

def forecast_future(model, last_window: np.ndarray, scaler: MinMaxScaler,
                    days: int = FORECAST_DAYS) -> np.ndarray:
    """
    Iterative multi-step forecast.
    We carry forward the last known feature values except Close,
    which is updated with each prediction.
    """
    current_window = last_window.copy()   # (window, n_features)
    preds_norm = []

    for _ in range(days):
        inp = current_window[np.newaxis, :, :]          # (1, window, n_features)
        pred_norm = model.predict(inp, verbose=0)[0, 0]
        preds_norm.append(pred_norm)

        # build next row: shift predicted Close into col 0, carry others
        next_row = current_window[-1].copy()
        next_row[0] = pred_norm
        current_window = np.vstack([current_window[1:], next_row])

    # denormalise Close only
    dummy = np.zeros((len(preds_norm), len(FEATURE_COLS)))
    dummy[:, 0] = preds_norm
    prices = scaler.inverse_transform(dummy)[:, 0]
    return prices


# ──────────────────────────────────────────────
#  EVALUATION
# ──────────────────────────────────────────────

def evaluate(model, X_test, y_test, scaler) -> dict:
    preds_norm = model.predict(X_test, verbose=0).flatten()

    dummy_pred = np.zeros((len(preds_norm), len(FEATURE_COLS)))
    dummy_pred[:, 0] = preds_norm
    dummy_true = np.zeros((len(y_test), len(FEATURE_COLS)))
    dummy_true[:, 0] = y_test

    preds_real = scaler.inverse_transform(dummy_pred)[:, 0]
    true_real  = scaler.inverse_transform(dummy_true)[:, 0]

    rmse = np.sqrt(mean_squared_error(true_real, preds_real))
    mae  = mean_absolute_error(true_real, preds_real)
    mape = np.mean(np.abs((true_real - preds_real) / true_real)) * 100

    return {
        "RMSE": round(rmse, 2),
        "MAE":  round(mae, 2),
        "MAPE": round(mape, 2),
        "test_preds": preds_real,
        "test_true":  true_real,
    }


# ──────────────────────────────────────────────
#  PLOTTING
# ──────────────────────────────────────────────

def plot_stock(ticker: str, df: pd.DataFrame, metrics: dict,
               future_dates, future_prices, history):
    color = COLORS.get(ticker, "#333")
    fig, axes = plt.subplots(3, 1, figsize=(14, 16))
    fig.suptitle(f"{ticker} — LSTM Stock Price Prediction", fontsize=16, fontweight="bold", y=0.98)
    fig.patch.set_facecolor("#F8F9FA")

    # ── Panel 1: Historical + forecast ──
    ax1 = axes[0]
    ax1.set_facecolor("#FFFFFF")
    hist_dates  = df.index[-len(metrics["test_true"]):]
    train_dates = df.index[WINDOW_SIZE:WINDOW_SIZE + len(metrics["test_preds"]) * 4 // 4]

    ax1.plot(df.index, df["Close"], color=color, linewidth=1.2, label="Actual close", alpha=0.9)
    ax1.plot(hist_dates, metrics["test_preds"], color="#E53935", linewidth=1.2,
             linestyle="--", label="Model fit (test)", alpha=0.85)
    ax1.plot(future_dates, future_prices, color="#FF6F00", linewidth=2.0,
             label=f"Forecast (next {FORECAST_DAYS}d)")

    conf = future_prices.std() * np.sqrt(np.arange(1, len(future_prices) + 1) / 30)
    ax1.fill_between(future_dates,
                     future_prices - 1.96 * conf,
                     future_prices + 1.96 * conf,
                     alpha=0.15, color="#FF6F00", label="95% confidence band")

    ax1.axvline(df.index[-1], color="gray", linestyle=":", linewidth=1)
    ax1.text(df.index[-1], ax1.get_ylim()[0], " Forecast starts", fontsize=8, color="gray")
    ax1.set_title("Historical price + 365-day forecast", fontsize=12)
    ax1.set_ylabel("Price (USD)")
    ax1.legend(fontsize=9, loc="upper left")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Training loss ──
    ax2 = axes[1]
    ax2.set_facecolor("#FFFFFF")
    ax2.plot(history.history["loss"],     color=color,     linewidth=1.5, label="Train loss")
    ax2.plot(history.history["val_loss"], color="#E53935", linewidth=1.5, label="Val loss", linestyle="--")
    ax2.set_title("Model training loss (Huber)", fontsize=12)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Technical indicators ──
    ax3 = axes[2]
    ax3.set_facecolor("#FFFFFF")
    recent = df.tail(180)
    ax3.plot(recent.index, recent["Close"],  color=color,     linewidth=1.2, label="Close")
    ax3.plot(recent.index, recent["SMA_20"], color="#43A047", linewidth=1.0, linestyle="--", label="SMA 20")
    ax3.plot(recent.index, recent["SMA_50"], color="#FB8C00", linewidth=1.0, linestyle="--", label="SMA 50")
    ax3.fill_between(recent.index, recent["BB_upper"], recent["BB_lower"],
                     alpha=0.1, color=color, label="Bollinger bands")
    ax3.set_title("Last 180 days — price + moving averages + Bollinger bands", fontsize=12)
    ax3.set_ylabel("Price (USD)")
    ax3.legend(fontsize=9, loc="upper left")
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax3.grid(True, alpha=0.3)

    # Metrics box
    m_text = (f"Test RMSE: ${metrics['RMSE']}   |   "
              f"MAE: ${metrics['MAE']}   |   "
              f"MAPE: {metrics['MAPE']}%")
    fig.text(0.5, 0.005, m_text, ha="center", fontsize=9,
             color="gray", style="italic")

    plt.tight_layout(rect=[0, 0.015, 1, 0.97])
    path = f"plots/{ticker}_prediction.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved chart → {path}")


def plot_comparison(all_results: dict):
    """Normalized comparison of all forecasts (base = 100 at today)."""
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#FFFFFF")

    for ticker, res in all_results.items():
        prices = res["future_prices"]
        base   = prices[0]
        norm   = (prices / base) * 100
        color  = COLORS.get(ticker, "#333")
        ax.plot(res["future_dates"], norm, color=color, linewidth=2, label=ticker)
        ax.annotate(f"{ticker} {norm[-1]:.1f}",
                    xy=(res["future_dates"][-1], norm[-1]),
                    xytext=(5, 0), textcoords="offset points",
                    color=color, fontsize=9, va="center")

    ax.axhline(100, color="gray", linestyle=":", linewidth=1)
    ax.set_title("1-year forecast comparison — all stocks (normalised, base = 100 today)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Normalised price (100 = current)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = "plots/comparison_forecast.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved comparison chart → {path}")


# ──────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────

def run(tickers=DEFAULT_TICKERS, days=FORECAST_DAYS):
    summary    = {}
    all_results = {}

    for ticker in tickers:
        print(f"\n{'='*55}")
        print(f"  Processing {ticker}")
        print(f"{'='*55}")

        # 1. Data
        df = fetch_stock_data(ticker)

        # 2. Preprocess
        X_train, X_test, y_train, y_test, scaler, scaled = prepare_data(df)
        print(f"  Train samples: {len(X_train)}  |  Test samples: {len(X_test)}")

        # 3. Build & train
        model = build_model(WINDOW_SIZE, len(FEATURE_COLS))
        print(f"  Training model ({EPOCHS} max epochs with early stopping)...")
        history = train_model(model, X_train, y_train, X_test, y_test)
        actual_epochs = len(history.history["loss"])
        print(f"  Stopped at epoch {actual_epochs}")

        # 4. Save model
        model.save(f"models/{ticker}_lstm.keras")

        # 5. Evaluate on test set
        metrics = evaluate(model, X_test, y_test, scaler)
        print(f"  RMSE: ${metrics['RMSE']}  |  MAE: ${metrics['MAE']}  |  MAPE: {metrics['MAPE']}%")

        # 6. Forecast future
        last_window    = scaled[-WINDOW_SIZE:]
        future_prices  = forecast_future(model, last_window, scaler, days)
        today          = df.index[-1]
        future_dates   = pd.bdate_range(start=today + timedelta(days=1), periods=days)

        # 7. Plot
        plot_stock(ticker, df, metrics, future_dates, future_prices, history)

        # 8. Collect results
        current_price = float(df["Close"].iloc[-1])
        year_end_price = float(future_prices[-1])
        pct_change = (year_end_price - current_price) / current_price * 100

        summary[ticker] = {
            "current_price":    round(current_price, 2),
            "predicted_1y":     round(year_end_price, 2),
            "pct_change":       round(pct_change, 2),
            "rmse":             metrics["RMSE"],
            "mae":              metrics["MAE"],
            "mape":             metrics["MAPE"],
            "training_epochs":  actual_epochs,
        }

        all_results[ticker] = {
            "future_dates":  future_dates,
            "future_prices": future_prices,
        }

    # 9. Comparison chart
    print(f"\n{'='*55}")
    print("  Generating comparison chart...")
    plot_comparison(all_results)

    # 10. Print summary table
    print(f"\n{'='*55}")
    print(f"  SUMMARY")
    print(f"{'='*55}")
    print(f"  {'Ticker':<8} {'Current':>10} {'1Y Pred':>10} {'Change':>10} {'MAPE':>8}")
    print(f"  {'-'*50}")
    for t, s in summary.items():
        arrow = "↑" if s["pct_change"] >= 0 else "↓"
        print(f"  {t:<8} ${s['current_price']:>9.2f} ${s['predicted_1y']:>9.2f} "
              f"{arrow}{abs(s['pct_change']):>8.1f}%  {s['mape']:>6.1f}%")

    # 11. Save JSON summary
    with open("data/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Full summary saved → data/summary.json")
    print(f"  All plots saved in → plots/")
    print(f"  Trained models in  → models/")
    print(f"\n  Done! ✓")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM Stock Price Prediction")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS,
                        help="Stock tickers (e.g. AAPL TSLA NVDA)")
    parser.add_argument("--days",    type=int, default=FORECAST_DAYS,
                        help="Forecast horizon in trading days (default 365)")
    args = parser.parse_args()
    run(tickers=args.tickers, days=args.days)
