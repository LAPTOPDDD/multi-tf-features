# -*- coding: utf-8 -*-
"""
Цель = некаузальная производная (БИНАРНАЯ: +1 / -1)
Фичи = каузальные + БИНАРНЫЕ
Только главный график: Target vs Prediction
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


# =====================================================================
# НАСТРОЙКИ
# =====================================================================
DATA_FILE = "EURUSD_H1_202401020000_202608271900.csv"
SIGMA = 10
LOOKBACK_MONTHS = 36
SYMBOL = "EURUSD"
N_TEST_BARS = 200
ROLL_WINDOW = 5


# =====================================================================
# ДАННЫЕ
# =====================================================================
def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    try:
        df = pd.read_csv(path, sep="\t")
    except:
        df = pd.read_csv(path)
    df.columns = [c.strip().strip("<>") for c in df.columns]
    col_map = {c: c.lower() for c in df.columns if c.lower() in ("open", "high", "low", "close")}
    if not col_map:
        raise ValueError(f"No columns: {list(df.columns)}")
    df = df.rename(columns=col_map)
    return df[["open", "high", "low", "close"]].astype(float).dropna()


# =====================================================================
# ГАУССИАНЫ
# =====================================================================
def gaussian_causal(data, sigma):
    n = len(data)
    result = np.zeros(n)
    q = min(int(4 * sigma) + 1, n)
    k = np.exp(-0.5 * (np.arange(q) / sigma) ** 2)
    k = k / k.sum()
    for i in range(n):
        if i < q:
            w = k[:i+1]
            result[i] = np.sum(data[:i+1] * w[::-1]) / w.sum()
        else:
            result[i] = np.sum(k * data[i-q+1:i+1])
    return result


def gaussian_noncausal(data, sigma):
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(data, sigma, mode="nearest")


# =====================================================================
# ФИЧИ (БИНАРНЫЕ каузальные)
# =====================================================================
def build_features(df, sigma):
    close = df["close"].values
    
    # Каузальная производная
    smoothed = gaussian_causal(close, sigma)
    deriv = np.zeros_like(close)
    deriv[1:] = smoothed[1:] - smoothed[:-1]
    deriv2 = np.zeros_like(close)
    deriv2[1:] = deriv[1:] - deriv[:-1]
    
    # Сглаживаем
    deriv_s = pd.Series(deriv).rolling(ROLL_WINDOW, min_periods=1).mean().values
    deriv2_s = pd.Series(deriv2).rolling(ROLL_WINDOW, min_periods=1).mean().values
    
    f = {}
    
    # --- Бинаризуем: > 0 -> +1, < 0 -> -1 ---
    
    f["deriv"] = np.where(deriv_s > 0, 1, -1)
    f["deriv2"] = np.where(deriv2_s > 0, 1, -1)
    
    # Сглаженные лаги (бинарные)
    for lag in range(1, 21):
        shifted = pd.Series(deriv_s).shift(lag).values
        f[f"deriv_lag{lag}"] = np.where(shifted > 0, 1, -1)
    for lag in range(1, 11):
        shifted = pd.Series(deriv2_s).shift(lag).values
        f[f"deriv2_lag{lag}"] = np.where(shifted > 0, 1, -1)
    
    # Returns (бинарные)
    for lag in range(1, 11):
        ret = df["close"].pct_change(lag).shift(1).values
        f[f"return_lag{lag}"] = np.where(ret > 0, 1, -1)
    
    # Свечные (бинарные)
    body = close - df["open"].values
    f["body_pos"] = np.where(body > 0, 1, -1)  # бычья свеча
    f["range_pos"] = np.where(df["high"].values - df["low"].values > 0, 1, -1)  # всегда +
    f["upper_shadow_pos"] = np.where(
        df["high"].values - np.maximum(df["open"].values, close) > 0, 1, -1
    )
    f["lower_shadow_pos"] = np.where(
        np.minimum(df["open"].values, close) - df["low"].values > 0, 1, -1
    )
    
    # MA (бинарные: close > MA)
    for w in [5, 10, 20]:
        ma = df["close"].rolling(w).mean().shift(1).values
        f[f"ma{w}_up"] = np.where(close > ma, 1, -1)
    
    # MA cross (бинарные)
    ma5 = df["close"].rolling(5).mean().shift(1).values
    ma20 = df["close"].rolling(20).mean().shift(1).values
    f["ma_cross"] = np.where(ma5 > ma20, 1, -1)
    
    # Волатильность (бинарная: выше медианы = высокая)
    vol = df["close"].rolling(20).std().shift(1).values
    vol_median = np.nanmedian(vol)
    f["vol_high"] = np.where(vol > vol_median, 1, -1)
    
    # High/Low (бинарные: close vs экстремумы)
    high10 = df["high"].rolling(10).max().shift(1).values
    low10 = df["low"].rolling(10).min().shift(1).values
    f["close_vs_high10"] = np.where(close > high10, 1, -1)
    f["close_vs_low10"] = np.where(close > low10, 1, -1)
    
    # Позиция в диапазоне 10 баров (бинарная)
    f["in_upper_half"] = np.where(close > (high10 + low10) / 2, 1, -1)
    
    # Momentum (бинарное)
    mom5 = close - df["close"].shift(5).values
    mom10 = close - df["close"].shift(10).values
    f["mom5_pos"] = np.where(mom5 > 0, 1, -1)
    f["mom10_pos"] = np.where(mom10 > 0, 1, -1)
    
    # RSI-подобное (бинарное)
    gain = df["close"].diff().clip(lower=0).rolling(14).mean().values
    loss = (-df["close"].diff().clip(upper=0)).rolling(14).mean().values
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_median = 50
    f["rsi_overbought"] = np.where(rsi > 70, 1, -1)  # перекупленность
    f["rsi_oversold"] = np.where(rsi < 30, 1, -1)    # перепроданность
    
    return pd.DataFrame(f, index=df.index)


# =====================================================================
# ЦЕЛЬ (БИНАРНАЯ)
# =====================================================================
def build_target(df, sigma):
    close = df["close"].values
    smoothed = gaussian_noncausal(close, sigma)
    deriv = np.gradient(smoothed)
    # БИНАРНАЯ: +1 если derivative > 0, -1 если < 0
    binary = np.where(deriv > 0, 1, -1)
    return binary


# =====================================================================
# ОБУЧЕНИЕ (БИНАРНАЯ КЛАССИФИКАЯ)
# =====================================================================
def train_model(X_train, y_train, X_val, y_val):
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.02,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 30,
        "verbose": -1,
        "seed": 42,
    }
    # Перекодируем: -1 -> 0, +1 -> 1
    y_train_bin = ((y_train + 1) // 2).astype(int)
    y_val_bin = ((y_val + 1) // 2).astype(int)
    
    train_set = lgb.Dataset(X_train, label=y_train_bin)
    val_set = lgb.Dataset(X_val, label=y_val_bin, reference=train_set)
    model = lgb.train(
        params, train_set, num_boost_round=2000,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)]
    )
    return model


# =====================================================================
# ГРАФИК
# =====================================================================
def plot_target_vs_prediction(target, prediction, n_test):
    n = min(len(target), len(prediction), n_test)
    x = np.arange(n)
    
    t = target[:n]
    pred_binary = np.where(prediction > 0.5, 1, -1)
    
    accuracy = np.mean(pred_binary == t)
    
    tp = np.sum((t == 1) & (pred_binary == 1))
    tn = np.sum((t == -1) & (pred_binary == -1))
    fp = np.sum((t == -1) & (pred_binary == 1))
    fn = np.sum((t == 1) & (pred_binary == -1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    fig, axes = plt.subplots(3, 1, figsize=(20, 12))
    
    # 1. Главный
    ax = axes[0]
    ax.step(x, t, where="post", color="#4CAF50", linewidth=2.5, label="Target (+1 / -1)", alpha=0.9)
    ax.step(x, pred_binary, where="post", color="#2196F3", linewidth=2.5, label="Prediction", alpha=0.9, linestyle="--")
    
    for i in range(n - 1):
        if pred_binary[i] == t[i]:
            ax.axvspan(x[i], x[i+1], alpha=0.15, color="#4CAF50")
        else:
            ax.axvspan(x[i], x[i+1], alpha=0.25, color="#F44336")
    
    ax.axhline(0, color="gray", linestyle="-", alpha=0.4)
    ax.set_ylabel("Value", fontsize=11)
    ax.set_title(
        f"BINARY TARGET vs PREDICTION (BINARY FEATURES)  |  {n} test bars\n"
        f"Accuracy={accuracy:.4f}  |  F1={f1:.4f}  |  Precision={precision:.4f}  |  Recall={recall:.4f}",
        fontsize=13, fontweight="bold"
    )
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["-1 (down)", "0", "+1 (up)"])
    ax.set_xlim(0, n)
    
    # 2. Вероятности
    ax = axes[1]
    ax.fill_between(x, 0, prediction, where=(prediction > 0.5), color="#4CAF50", alpha=0.4)
    ax.fill_between(x, 0, prediction, where=(prediction <= 0.5), color="#F44336", alpha=0.4)
    ax.plot(x, prediction, color="#2196F3", linewidth=1.5, alpha=0.9)
    ax.axhline(0.5, color="black", linestyle="-", alpha=0.8, linewidth=1.5)
    ax.set_ylabel("Probability", fontsize=11)
    ax.set_title("Prediction Probability", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    
    # 3. Confusion Matrix
    ax = axes[2]
    ax.axis("off")
    
    total = tp + tn + fp + fn
    matrix_text = (
        f"CONFUSION MATRIX\n"
        f"{'='*40}\n\n"
        f"                  Predicted\n"
        f"              -1        +1\n"
        f"         +--------+--------+\n"
        f" Actual  |  TN={tn:4d}  |  FP={fp:4d}  |\n"
        f"   -1     | {tn/total*100:5.1f}%  | {fp/total*100:5.1f}%  |\n"
        f"         +--------+--------+\n"
        f" Actual  |  FN={fn:4d}  |  TP={tp:4d}  |\n"
        f"   +1     | {fn/total*100:5.1f}%  | {tp/total*100:5.1f}%  |\n"
        f"         +--------+--------+\n\n"
        f"Total: {total}  |  Correct: {tp+tn}  |  Wrong: {fp+fn}\n"
        f"Accuracy: {accuracy:.4f}  |  F1: {f1:.4f}"
    )
    
    ax.text(0.35, 0.5, matrix_text, fontsize=13, fontfamily="monospace",
            verticalalignment="center", horizontalalignment="center",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    
    plt.tight_layout()
    
    out_path = os.path.join(os.getcwd(), "result.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Chart: {out_path}")
    
    return accuracy, precision, recall, f1, tp, tn, fp, fn


# =====================================================================
# ВАЖНОСТЬ ФИЧ
# =====================================================================
def plot_importance(model, features):
    imp = pd.DataFrame({
        "feature": features.columns,
        "importance": model.feature_importance()
    }).sort_values("importance", ascending=False)
    
    imp.to_csv("importance.csv", index=False)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    top = imp.head(20)
    colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(top)))
    ax.barh(range(len(top)), top["importance"].values, color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"].values)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance (Binary Features)")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    
    out_path = os.path.join(os.getcwd(), "importance.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Chart: {out_path}")
    
    return imp


# =====================================================================
# ГЛАВНАЯ
# =====================================================================
def main():
    print("=" * 70)
    print("  BINARY TARGET + BINARY FEATURES")
    print("  (non-causal derivative -> +1 / -1)")
    print("=" * 70)
    
    df = load_data(DATA_FILE)
    bars = 24 * 30 * LOOKBACK_MONTHS
    if len(df) > bars:
        df = df.iloc[-bars:].reset_index(drop=True)
    
    n = len(df)
    train_end = int(n * 0.70)
    
    print(f"  File: {DATA_FILE}")
    print(f"  Total: {n} candles")
    print(f"  Sigma: {SIGMA}, Smooth: {ROLL_WINDOW}")
    print(f"  Test bars: {N_TEST_BARS}")
    print(f"  Train: 0:{train_end}  |  Test: {train_end}:{train_end + N_TEST_BARS}")
    
    if not HAS_LGB:
        print("  Install: pip install lightgbm")
        return
    
    # Цель (бинарная)
    print("\n  Building BINARY TARGET...")
    target = build_target(df, SIGMA)
    n_pos = np.sum(target == 1)
    neg = np.sum(target == -1)
    print(f"  Class +1: {n_pos} ({n_pos/len(target)*100:.1f}%)")
    
    
    
    # Фичи (бинарные)
    print("\n  Building BINARY FEATURES...")
    features = build_features(df, SIGMA)
    n_features = features.shape[1]
    print(f"  Features: {n_features}")
    
    # Проверка: все фичи должны быть +1 или -1
    unique_vals = features.apply(lambda col: col.unique())
    print(f"  Feature values: unique per column (sample)")
    
    # Выравнивание
    X = features.dropna()
    y_full = pd.Series(target, index=df.index)
    y = y_full.loc[X.index]
    
    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    X_val = X.iloc[train_end: train_end + N_TEST_BARS]
    y_val = y.iloc[train_end: train_end + N_TEST_BARS]
    
    print(f"\n  SPLIT:")
    print(f"  Train: {len(X_train)}")
    print(f"  Val:   {len(X_val)}")
    
    # Обучение
    print("\n  Training (binary classification)...")
    model = train_model(X_train, y_train, X_val, y_val)
    print(f"  Trees: {model.num_trees()}")
    
    # Предсказание
    print("\n  Predicting...")
    proba = model.predict(X_val)
    target_vals = y_val.values
    
    # Графики
    print("\n  Building chart...")
    accuracy, precision, recall, f1, tp, tn, fp, fn = plot_target_vs_prediction(
        target_vals, proba, N_TEST_BARS
    )
    importance = plot_importance(model, features)
    
    print("\n" + "=" * 70)
    print("  RESULTS (BINARY FEATURES)")
    print("=" * 70)
    print(f"  Model      : LightGBM Binary (sigma={SIGMA})")
    print(f"  Features   : {n_features} (все бинарные: +1 / -1)")
    print(f"  Trees      : {model.num_trees()}")
    print(f"  Test       : {len(X_val)}")
    print("-" * 70)
    print(f"  Baseline (random)    : 0.5000")
    print(f"  Model accuracy       : {accuracy:.4f}")
    print(f"  Improvement          : {accuracy - 0.5:+.4f}")
    print(f"  F1 Score             : {f1:.4f}")
    print(f"  Precision            : {precision:.4f}")
    print(f"  Recall               : {recall:.4f}")
    print("-" * 70)
    print(f"  Confusion Matrix:")
    print(f"    TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print("-" * 70)

    print("\n  TOP-15 FEATURES:")
    for _, row in importance.head(15).iterrows():
        bar = "█" * int(row["importance"] / importance["importance"].max() * 20)
        print(f"    {row['feature']:28s} {bar} {row['importance']:.0f}")

    print("\n  Done!")


if __name__ == "__main__":
    main()