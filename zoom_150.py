"""
Детальный zoom-график: целевой сигнал vs предсказание модели
на участке 150 баров из тестовой выборки.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ==================== 1. ДАННЫЕ (синтетические, как в основном скрипте) ====================
n = 5000
dates = pd.date_range(start='2025-01-01', periods=n, freq='5min')
close = np.cumsum(np.random.randn(n) * 0.0005) + 1.10
high = close + np.abs(np.random.randn(n) * 0.0005)
low = close - np.abs(np.random.randn(n) * 0.0005)
open_ = np.roll(close, 1) + np.random.randn(n) * 0.0001
open_[0] = close[0]
tick_vol = np.random.randint(100, 1000, n).astype(float)

m5 = pd.DataFrame({
    'time': dates, 'open': open_, 'high': high, 'low': low, 'close': close,
    'tick_volume': tick_vol, 'real_volume': tick_vol, 'spread': np.random.randint(1, 10, n)
})
m5.set_index('time', inplace=True)

# ==================== 2. ЦЕЛЕВОЙ СИГНАЛ (Ground Truth) ====================
def gaussian_derivative_target(close, sigma=5):
    smoothed = gaussian_filter1d(np.log(close), sigma=sigma, mode='nearest')
    derivative = np.gradient(smoothed)
    return pd.Series(derivative, index=close.index)

SIGMA = 5
ideal_target = gaussian_derivative_target(m5['close'], SIGMA)
target_mean = ideal_target.rolling(500).mean()
target_std = ideal_target.rolling(500).std()
target_norm = (ideal_target - target_mean) / target_std

# ==================== 3. ФИЧИ + ЛАГИ 1-5 ====================
def add_real_features(df):
    df = df.copy()
    c = df['close']
    h = df['high']
    l = df['low']
    o = df['open']
    
    df['open_price'] = o
    df['prev_close'] = c.shift(1)
    df['prev_high'] = h.shift(1)
    df['prev_low'] = l.shift(1)
    df['prev_range'] = (h - l).shift(1)
    df['prev_body'] = (c - o).shift(1)
    
    for period in [5, 10, 20, 50, 100]:
        df[f'sma_{period}'] = c.rolling(period).mean().shift(1)
        df[f'ema_{period}'] = c.ewm(span=period, adjust=False).mean().shift(1)
    
    def rsi_causal(s, period=14):
        delta = s.diff()
        gain = (delta.clip(lower=0)).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - 100 / (1 + rs)
    
    df['rsi_14'] = rsi_causal(c, 14).shift(1)
    df['rsi_21'] = rsi_causal(c, 21).shift(1)
    
    tr1 = h - l
    tr2 = (h - c.shift()).abs()
    tr3 = (l - c.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean().shift(1)
    
    df['vol_10'] = c.pct_change().rolling(10).std().shift(1)
    df['vol_20'] = c.pct_change().rolling(20).std().shift(1)
    
    for period in [5, 10, 20]:
        df[f'momentum_{period}'] = (c / c.shift(period) - 1).shift(1)
    
    df['close_to_sma20'] = (c / df['sma_20'] - 1)
    df['close_to_sma50'] = (c / df['sma_50'] - 1)
    df['hl_ratio'] = ((h - l) / c).shift(1)
    df['co_ratio'] = ((c - o) / o).shift(1)
    
    # ЛАГИ 1-5
    base_features = df.columns.tolist()
    for feature in base_features:
        for lag in range(1, 6):
            col_name = f'{feature}_lag{lag}'
            df[col_name] = df[feature].shift(lag)
    
    return df

features = add_real_features(m5)
features = features.dropna()

# ==================== 4. ПОДГОТОВКА ДАННЫХ ====================
target_aligned = target_norm.loc[features.index]
y = target_aligned.shift(-1).dropna()
X = features.loc[y.index]

split_idx = int(len(X) * 0.8)
test_size = 500

X_train = X.iloc[:split_idx]
X_test = X.iloc[-test_size:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[-test_size:]

# ==================== 5. ОБУЧЕНИЕ ====================
model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    eval_metric='rmse',
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
)

y_pred = model.predict(X_test)

# ==================== 6. ZOOM НА 150 БАРОВ ====================
# Берём первые 150 баров из теста для детального сравнения
ZOOM_BARS = 150
idx = slice(0, ZOOM_BARS)

y_test_zoom = y_test.iloc[idx]
y_pred_zoom = y_pred[idx]
price_zoom = m5['close'].loc[y_test_zoom.index]

fig, axes = plt.subplots(3, 1, figsize=(16, 12))

# --- Панель 1: Целевой сигнал vs Предсказание ---
ax1 = axes[0]
ax1.plot(y_test_zoom.index, y_test_zoom.values, label='Целевой сигнал (Ground Truth)',
         color='steelblue', linewidth=2, alpha=0.9)
ax1.plot(y_test_zoom.index, y_pred_zoom, label='Предсказание модели',
         color='crimson', linewidth=1.5, alpha=0.85)
ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax1.set_title(f'Целевой сигнал vs Предсказание (первые {ZOOM_BARS} баров теста)', fontsize=13)
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3)

# Локальная корреляция на этом участке
local_corr = np.corrcoef(y_test_zoom.values, y_pred_zoom)[0, 1]
ax1.text(0.02, 0.95, f'Локальная корреляция: {local_corr:.3f}',
         transform=ax1.transAxes, fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# --- Панель 2: Ошибка (остатки) ---
ax2 = axes[1]
error = y_pred_zoom - y_test_zoom.values
ax2.plot(y_test_zoom.index, error, color='darkorange', linewidth=1, alpha=0.8)
ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax2.fill_between(y_test_zoom.index, 0, error, alpha=0.25, color='darkorange')
ax2.set_title('Ошибка предсказания (Pred - Target)', fontsize=13)
ax2.grid(True, alpha=0.3)

mse_local = np.mean(error**2)
ax2.text(0.02, 0.95, f'Локальный MSE: {mse_local:.6f}',
         transform=ax2.transAxes, fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

# --- Панель 3: Цена (контекст) ---
ax3 = axes[2]
ax3.plot(price_zoom.index, price_zoom.values, color='green', linewidth=1.2, alpha=0.8)
ax3.set_title('Цена EURUSD (контекст)', fontsize=13)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chart_zoom_150.png', dpi=150)
print(f"\n✅ Сохранён: chart_zoom_150.png")
print(f"Локальная корреляция на {ZOOM_BARS} барах: {local_corr:.4f}")
print(f"Локальный MSE: {mse_local:.8f}")
print(f"Глобальная корреляция на всём тесте: {np.corrcoef(y_test, y_pred)[0,1]:.4f}")
plt.show()
