# -*- coding: utf-8 -*-
"""
Цель = некаузальная производная (oscillates around zero)
Фичи = все каузальные
Ожидание: выход модели повторяет цель
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
FORWARD = 0      # Цель = производная СЕЙЧАС, не будущее
LOOKBACK_MONTHS = 36
SYMBOL = "EURUSD"
N_TEST_BARS = 100   # Только 100 тестовых баров


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
# КАУЗАЛЬНЫЙ ГАУСС (только прошлое)
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


# =====================================================================
# НЕКАУЗАЛЬНЫЙ ГАУСС (полное окно — для цели)
# =====================================================================
def gaussian_noncausal(data, sigma):
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(data, sigma, mode="nearest")


# =====================================================================
# КАУЗАЛЬНЫЕ ФИЧИ
# =====================================================================
def build_causal_features(df, sigma):
    close = df["close"].values
    smoothed = gaussian_causal(close, sigma)
    
    # Производная (каузальная, backward)
    deriv = np.zeros_like(close)
    deriv[1:] = smoothed[1:] - smoothed[:-1]
    
    # Вторая производная (каузальная)
    deriv2 = np.zeros_like(close)
    deriv2[1:] = deriv[1:] - deriv[:-1]
    
    features = {}
    body = close - df["open"].values
    
    features["body"] = body
    features["range"] = df["high"].values - df["low"].values
    features["upper_shadow"] = df["high"].values - np.maximum(df["open"].values, close)
    features["lower_shadow"] = np.minimum(df["open"].values, close) - df["low"].values
    features["is_bullish"] = (body > 0).astype(float)
    
    features["deriv_causal"] = deriv
    features["deriv2_causal"] = deriv2
    
    # Лаги
    for lag in range(1, 21):
        features[f"deriv_lag{lag}"] = pd.Series(deriv).shift(lag).values
        features[f"deriv2_lag{lag}"] = pd.Series(deriv2).shift(lag).values
    
    # Returns
    for lag in range(1, 11):
        features[f"return_lag{lag}"] = df["close"].pct_change(lag).shift(1).values
    
    # MA
    features["ma_5"] = df["close"].rolling(5).mean().shift(1).values
    features["ma_10"] = df["close"].rolling(10).mean().shift(1).values
    features["ma_20"] = df["close"].rolling(20).mean().shift(1).values
    features["volatility"] = df["close"].rolling(20).std().shift(1).values
    
    # MA crossover
    ma5 = df["close"].rolling(5).mean().shift(1).values
    ma20 = df["close"].rolling(20).mean().shift(1).values
    features["ma_cross"] = ma5 - ma20
    
    # High/Low
    features["high_10"] = df["high"].rolling(10).max().shift(1).values
    features["low_10"] = df["low"].rolling(10).min().shift(1).values
    features["close_vs_high10"] = close - features["high_10"]
    features["close_vs_low10"] = close - features["low_10"]
    
    return pd.DataFrame(features, index=df.index)


# =====================================================================
# ЦЕЛЬ: некаузальная производная
# =====================================================================
def build_target(df, sigma):
    close = df["close"].values
    # Некаузальное сглаживание
    smoothed = gaussian_noncausal(close, sigma)
    # Центральная разность (использует будущее!)
    deriv = np.gradient(smoothed)
    return deriv


# =====================================================================
# ОБУЧЕНИЕ
# =====================================================================
def train_model(X_train, y_train, X_val, y_val):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.03,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
    }
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
    return lgb.train(
        params, train_set, num_boost_round=1000,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )


# =====================================================================
# ГРАФИК 1: Цель vs Предсказание
# =====================================================================
def plot_target_vs_prediction(target, prediction, n_test):
    n = min(len(target), len(prediction), n_test)
    x = np.arange(n)
    
    target_n = target[:n]
    pred_n = prediction[:n]
    
    # Метрики
    rmse = np.sqrt(np.mean((target_n - pred_n) ** 2))
    mae = np.mean(np.abs(target_n - pred_n))
    corr = np.corrcoef(target_n, pred_n)[0, 1]
    
    # Нормализуем для сравнения
    t_std = target_n.std()
    t_mean = target_n.mean()
    target_norm = (target_n - t_mean) / (t_std + 1e-10)
    pred_norm = (pred_n - t_mean) / (t_std + 1e-10)
    
    fig, axes = plt.subplots(4, 1, figsize=(20, 16))
    
    # 1. Цель (oscillates around zero)
    ax1 = axes[0]
    ax1.plot(x, target_n, color="#4CAF50", linewidth=1.5, label="Target (non-causal derivative)")
    ax1.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax1.fill_between(x, 0, target_n, where=(target_n > 0), color="#4CAF50", alpha=0.2)
    ax1.fill_between(x, 0, target_n, where=(target_n < 0), color="#F44336", alpha=0.2)
    ax1.set_ylabel("Target Value", fontsize=10)
    ax1.set_title(
        f"TARGET vs PREDICTION  |  {n} test bars  |  sigma={SIGMA}\n"
        f"RMSE={rmse:.6f}  |  MAE={mae:.6f}  |  Correlation={corr:.3f}",
        fontsize=12, fontweight="bold"
    )
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, n)
    
    # 2. Предсказание модели
    ax2 = axes[1]
    ax2.plot(x, pred_n, color="#2196F3", linewidth=1.5, label="Model Prediction")
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax2.fill_between(x, 0, pred_n, where=(pred_n > 0), color="#2196F3", alpha=0.2)
    ax2.fill_between(x, 0, pred_n, where=(pred_n < 0), color="#F44336", alpha=0.2)
    ax2.set_ylabel("Prediction", fontsize=10)
    ax2.set_title("Model Output (causal features only)", fontsize=11)
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, n)
    
    # 3. Цель + Предсказание вместе
    ax3 = axes[2]
    ax3.plot(x, target_n, color="#4CAF50", linewidth=2, label="Target", alpha=0.9)
    ax3.plot(x, pred_n, color="#FF9800", linewidth=1.5, label="Prediction", alpha=0.8)
    ax3.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax3.set_ylabel("Value", fontsize=10)
    ax3.set_title("Overlay: Target (green) vs Prediction (orange)", fontsize=11)
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, n)
    
    # 4. Ошибка
    ax4 = axes[3]
    error = pred_n - target_n
    ax4.plot(x, error, color="#9C27B0", linewidth=1.2, label=f"Error (Prediction - Target)")
    ax4.fill_between(x, 0, error, color="#9C27B0", alpha=0.3)
    ax4.axhline(0, color="gray", linestyle="--", alpha=0.7)
    ax4.axhline(mae, color="red", linestyle="--", alpha=0.7, label=f"MAE={mae:.6f}")
    ax4.axhline(-mae, color="red", linestyle="--", alpha=0.7)
    ax4.set_ylabel("Error", fontsize=10)
    ax4.set_xlabel("Bar", fontsize=10)
    ax4.set_title("Prediction Error", fontsize=11)
    ax4.legend(loc="upper right")
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, n)
    
    plt.tight_layout()
    plt.savefig("target_vs_prediction.png", dpi=130, bbox_inches="tight")
    print("  Chart: target_vs_prediction.png")
    
    return rmse, mae, corr


# =====================================================================
# ГРАФИК 2: Распределение
# =====================================================================
def plot_distribution(target, prediction, n_test):
    n = min(len(target), len(prediction), n_test)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Гистограмма
    axes[0].hist(target[:n], bins=40, color="#4CAF50", alpha=0.6, label="Target")
    axes[0].hist(prediction[:n], bins=40, color="#2196F3", alpha=0.6, label="Prediction")
    axes[0].axvline(0, color="gray", linestyle="--")
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution: Target vs Prediction")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Scatter
    axes[1].scatter(target[:n], prediction[:n], alpha=0.4, s=10, c="#2196F3")
    lim = (min(target[:n].min(), prediction[:n].min()), max(target[:n].max(), prediction[:n].max()))
    axes[1].plot(lim, lim, "r--", linewidth=2, label="Perfect fit")
    axes[1].set_xlabel("Target")
    axes[1].set_ylabel("Prediction")
    axes[1].set_title("Scatter: Target vs Prediction")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # QQ-подобный
    t_sorted = np.sort(target[:n])
    p_sorted = np.sort(prediction[:n])
    axes[2].plot(t_sorted, p_sorted, color="#2196F3", linewidth=1.5)
    axes[2].plot(lim, lim, "r--", linewidth=2, label="Perfect")
    axes[2].set_xlabel("Target Quantiles")
    axes[2].set_ylabel("Prediction Quantiles")
    axes[2].set_title("Q-Q Plot")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("distribution.png", dpi=130, bbox_inches="tight")
    print("  Chart: distribution.png")


# =====================================================================
# ГРАФИК 3: Важность фич
# =====================================================================
def plot_importance(model, features):
    imp = pd.DataFrame({
        "feature": features.columns,
        "importance": model.feature_importance()
    }).sort_values("importance", ascending=False)
    
    imp.to_csv("feature_importance.csv", index=False)
    print("  CSV: feature_importance.csv")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    top = imp.head(20)
    colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(top)))
    ax.barh(range(len(top)), top["importance"].values, color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"].values)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance  |  sigma={SIGMA}")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=130, bbox_inches="tight")
    print("  Chart: feature_importance.png")
    
    return imp


# =====================================================================
# ГРАФИК 4: Сравнение с каузальной производной
# =====================================================================
def plot_derivative_comparison(deriv_causal, deriv_noncausal, prediction, n_test):
    n = min(len(deriv_causal), len(deriv_noncausal), len(prediction), n_test)
    x = np.arange(n)
    
    fig, axes = plt.subplots(3, 1, figsize=(20, 10), sharex=True)
    
    # Некаузальная (цель) vs каузальная
    axes[0].plot(x, deriv_noncausal[:n], color="#4CAF50", linewidth=1.5, label="Non-causal (TARGET)", alpha=0.8)
    axes[0].plot(x, deriv_causal[:n], color="#FF9800", linewidth=1.5, label="Causal derivative", alpha=0.6)
    axes[0].axhline(0, color="gray", linestyle="--", alpha=0.5)
    axes[0].set_ylabel("Derivative")
    axes[0].set_title("Non-causal vs Causal Derivative")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Корреляция каузальной с целью
    corr_c = np.corrcoef(deriv_causal[:n], deriv_noncausal[:n])[0, 1]
    axes[1].plot(x, deriv_noncausal[:n], color="#4CAF50", linewidth=2, label=f"Non-causal (target)", alpha=0.9)
    axes[1].plot(x, deriv_causal[:n], color="#FF9800", linewidth=1.5, label=f"Causal (corr={corr_c:.3f})", alpha=0.7)
    axes[1].axhline(0, color="gray", linestyle="--", alpha=0.5)
    axes[1].set_ylabel
     # Корреляция каузальной с целью
    corr_c = np.corrcoef(deriv_causal[:n], deriv_noncausal[:n])[0, 1]
    corr_p = np.corrcoef(prediction[:n], deriv_noncausal[:n])[0, 1]
    axes[1].plot(x, deriv_noncausal[:n], color="#4CAF50", linewidth=2, label=f"Non-causal (target)", alpha=0.9)
    axes[1].plot(x, deriv_causal[:n], color="#FF9800", linewidth=1.5, label=f"Causal (corr={corr_c:.3f})", alpha=0.7)
    axes[1].axhline(0, color="gray", linestyle="--", alpha=0.5)
    axes[1].set_ylabel("Derivative")
    axes[1].set_title("Can Causal Derivative Replicate Non-causal?")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Предсказание vs цель
    axes[2].plot(x, deriv_noncausal[:n], color="#4CAF50", linewidth=2, label="Non-causal (TARGET)", alpha=0.8)
    axes[2].plot(x, prediction[:n], color="#2196F3", linewidth=1.5, label=f"Prediction (corr={corr_p:.3f})", alpha=0.7)
    axes[2].axhline(0, color="gray", linestyle="--", alpha=0.5)
    axes[2].set_ylabel("Value")
    axes[2].set_xlabel("Bar")
    axes[2].set_title("Model Prediction vs Non-causal Target")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("derivative_comparison.png", dpi=130, bbox_inches="tight")
    print("  Chart: derivative_comparison.png")


# =====================================================================
# ГЛАВНАЯ
# =====================================================================
def main():
    print("=" * 70)
    print("  CAUSAL FEATURES -> NON-CAUSAL TARGET")
    print("=" * 70)
    
    df = load_data(DATA_FILE)
    bars = 24 * 30 * LOOKBACK_MONTHS
    if len(df) > bars:
        df = df.iloc[-bars:].reset_index(drop=True)
    
    n = len(df)
    train_end = int(n * 0.70)
    
    print(f"  File: {DATA_FILE}")
    print(f"  Total: {n} candles")
    print(f"  Sigma: {SIGMA}")
    print(f"  Test bars: {N_TEST_BARS}")
    
    if not HAS_LGB:
        print("  Install: pip install lightgbm")
        return
    
    # Цель: некаузальная производная
    print("\n  Building TARGET (non-causal derivative)...")
    target = build_target(df, SIGMA)
    target_std = np.std(target)
    target_mean = np.mean(target)
    print(f"  Target mean: {target_mean:.8f} (should be ~0)")
    print(f"  Target std:  {target_std:.8f}")
    
    # Каузальные фичи
    print("\n  Building CAUSAL features...")
    features = build_causal_features(df, SIGMA)
    n_features = features.shape[1]
    print(f"  Features: {n_features}")
    
    # Разделение
    X = features.dropna()
    y = pd.Series(target, index=df.index).loc[X.index]
    
    X_train = X.iloc[:train_end]
    y_train = y.loc[X_train.index]
    X_val = X.iloc[train_end:]
    y_val = y.loc[X_val.index]
    
    # Только последние N_TEST_BARS для валидации
    X_val = X_val.iloc[-N_TEST_BARS:]
    y_val = y_val.iloc[-N_TEST_BARS:]
    
    print(f"  Train: {len(X_train)}, Val/Test: {len(X_val)}")
    
    # Обучение
    print("\n  Training...")
    model = train_model(X_train, y_train, X_val, y_val)
    print(f"  Trees: {model.num_trees()}, Best iter: {model.best_iteration}")
    
    # Предсказание
    print("\n  Predicting on test bars...")
    prediction = model.predict(X_val)
    target_vals = y_val.values
    
    # Каузальная производная (для сравнения)
    deriv_causal = gaussian_causal(df["close"].values, SIGMA)
    deriv_causal = pd.Series(deriv_causal, index=df.index).loc[y_val.index].values
    
    # Графики
    print("\n  Building charts...")
    rmse, mae, corr = plot_target_vs_prediction(target_vals, prediction, N_TEST_BARS)
    plot_distribution(target_vals, prediction, N_TEST_BARS)
    plot_derivative_comparison(deriv_causal, target_vals, prediction, N_TEST_BARS)
    importance = plot_importance(model, features)
    
    # Результаты
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print(f"  MODEL        : LightGBM (sigma={SIGMA})")
    print(f"  FEATURES    : {n_features} (all causal)")
    print(f"  TARGET      : Non-causal derivative")
    print(f"  TREES       : {model.num_trees()}")
    print("-" * 70)
    print(f"  RMSE        : {rmse:.8f}")
    print(f"  MAE         : {mae:.8f}")
    print(f"  CORRELATION : {corr:.4f}")
    print("-" * 70)
    
    # Сравнение: каузальная vs цель
    corr_causal = np.corrcoef(deriv_causal, target_vals)[0, 1]
    print(f"  Causal deriv correlation with target: {corr_causal:.4f}")
    print(f"  Model prediction correlation:         {corr:.4f}")
    print(f"  Improvement:                          {corr - corr_causal:+.4f}")
    print("-" * 70)
    
    print("\n  TOP-10 FEATURES:")
    for _, row in importance.head(10).iterrows():
        bar = "█" * int(row["importance"] / importance["importance"].max() * 20)
        print(f"    {row['feature']:25s} {bar} {row['importance']:.0f}")
    
    print("\n  Done!")


if __name__ == "__main__":
    main()