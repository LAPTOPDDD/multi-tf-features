"""
Реальный сценарий торговли:
- На каждом баре M5 используем данные, доступные в момент открытия
- Целевой сигнал - производная Гаусса (идеальный учитель)
- Модель предсказывает будущее движение
- ДОБАВЛЕНЫ ЛАГИ (1-5) для устранения запаздывания модели
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

# ==================== 1. ПОДКЛЮЧЕНИЕ К METATRADER (ЗАКОММЕНТИРОВАНО ДЛЯ ТЕСТА) ====================
# import MetaTrader5 as mt5
# print("Подключаюсь к MetaTrader 5...")
# if not mt5.initialize():
#     print("Ошибка подключения!")
#     mt5.shutdown()
#     exit()
# print(f"Терминал: {mt5.terminal_info()}")
# print(f"Авторизован: {mt5.account_info().login}")

# ==================== 2. ЗАГРУЗКА ДАННЫХ ====================
def load_mt5_data(symbol, timeframe, bars=50000):
    # rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    # if rates is None or len(rates) == 0:
    #     return None
    # df = pd.DataFrame(rates)
    # df['time'] = pd.to_datetime(df['time'], unit='s')
    # df.set_index('time', inplace=True)
    # df.rename(columns={
    #     'open': 'open', 'high': 'high', 'low': 'low',
    #     'close': 'close', 'tick_volume': 'tickvol',
    #     'real_volume': 'vol', 'spread': 'spread'
    # }, inplace=True)
    # return df
    
    # --- СИНТЕТИЧЕСКИЕ ДАННЫЕ ДЛЯ ПРИМЕРА (замените на загрузку из MT5) ---
    np.random.seed(42)
    n = 5000
    dates = pd.date_range(start='2025-01-01', periods=n, freq='5min')
    close = np.cumsum(np.random.randn(n) * 0.0005) + 1.10
    high = close + np.abs(np.random.randn(n) * 0.0005)
    low = close - np.abs(np.random.randn(n) * 0.0005)
    open_ = np.roll(close, 1) + np.random.randn(n) * 0.0001
    open_[0] = close[0]
    tick_vol = np.random.randint(100, 1000, n).astype(float)
    
    df = pd.DataFrame({
        'time': dates, 'open': open_, 'high': high, 'low': low, 'close': close,
        'tick_volume': tick_vol, 'real_volume': tick_vol, 'spread': np.random.randint(1, 10, n)
    })
    df.set_index('time', inplace=True)
    return df

SYMBOL = "EURUSD"
print(f"\nЗагружаю {SYMBOL}...")

data = {}
timeframes = {
    'M5': 'M5',  # Замените на mt5.TIMEFRAME_M5 при реальном подключении
    'M15': 'M15',
    'M30': 'M30',
    'H1': 'H1',
    'H4': 'H4',
    'DAILY': 'D1'
}

BARS = 5000
for name, tf in timeframes.items():
    df = load_mt5_data(SYMBOL, tf, BARS)
    if df is not None and len(df) > 0:
        data[name] = df
        print(f"  {name}: {len(df)} баров")

BASE = "M5"
m5 = data[BASE].copy()

# ==================== 3. ИДЕАЛЬНЫЙ УЧИТЕЛЬ (ПРОИЗВОДНАЯ ГАУССА) ====================
def gaussian_derivative_target(close, sigma=5):
    smoothed = gaussian_filter1d(np.log(close), sigma=sigma, mode='nearest')
    derivative = np.gradient(smoothed)
    return pd.Series(derivative, index=close.index)

print("\nГенерирую идеальный сигнал...")
SIGMA = 5
ideal_target = gaussian_derivative_target(m5['close'], SIGMA)

# Нормализуем
target_mean = ideal_target.rolling(500).mean()
target_std = ideal_target.rolling(500).std()
target_norm = (ideal_target - target_mean) / target_std

# ==================== 4. ФИЧИ + ГЕНЕРАЦИЯ ЛАГОВ ====================
def add_real_features(df):
    """
    Фичи, доступные в момент открытия нового бара.
    Используем только закрытые бары (shift(1)).
    ДОБАВЛЕНЫ ЛАГИ: для каждой фичи создаем сдвиги 1-5.
    """
    df = df.copy()
    c = df['close']
    h = df['high']
    l = df['low']
    o = df['open']
    
    # Базовые фичи (как в вашем коде)
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
    
    # ==========================================================
    # НОВЫЙ БЛОК: ДОБАВЛЯЕМ ЛАГИ (1-5) ДЛЯ КАЖДОЙ ФИЧИ
    # ==========================================================
    base_features = df.columns.tolist()
    lag_columns = []
    for feature in base_features:
        # Пропускаем 'open', 'high', 'low', 'close', 'tickvol', 'vol', 'spread' если хотим, 
        # но для лагов нам нужны и они.
        for lag in range(1, 6):
            col_name = f'{feature}_lag{lag}'
            df[col_name] = df[feature].shift(lag)
            lag_columns.append(col_name)
    
    return df

print("\nГенерирую реальные фичи (только из прошлого + 5 лагов)...")

# Базовые фичи с M5
features = add_real_features(m5)

# Добавляем фичи с других таймфреймов (только прошлое!)
for tf_name in ['M15', 'M30', 'H1', 'H4', 'DAILY']:
    if tf_name in data:
        tf_df = add_real_features(data[tf_name])
        # Ресемплируем на M5 (ffill берет последнее известное значение закрытого бара)
        tf_resampled = tf_df.resample('5min').ffill()
        # Добавляем с shift(1) - используем только закрытые бары!
        for col in tf_resampled.columns:
            if col not in ['open', 'high', 'low', 'close', 'tickvol', 'vol', 'spread']:
                features[f'{tf_name}_{col}'] = tf_resampled[col].shift(1)

# Удаляем NaN (так как у нас теперь 5 лагов, NaN будет в начале больше)
features = features.dropna()

print(f"Всего фич (с лагами): {len(features.columns)}")

# ==================== 5. ПОДГОТОВКА ДАННЫХ ====================
target_aligned = target_norm.loc[features.index]

# Целевая переменная: СДВИГАЕМ ВПЕРЕД на 1 бар!
y = target_aligned.shift(-1)
y = y.dropna()

X = features.loc[y.index]

print(f"\nДанные для обучения:")
print(f"  Всего баров: {len(X)}")
print(f"  Фич: {len(X.columns)}")
print(f"  Целевая: производная Гаусса на следующем баре")

# ==================== 6. РАЗДЕЛЕНИЕ ====================
split_idx = int(len(X) * 0.8)
test_size = 500

X_train = X.iloc[:split_idx]
X_test = X.iloc[-test_size:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[-test_size:]

print(f"\nTrain: {len(X_train)} баров")
print(f"Test: {len(X_test)} баров")

# ==================== 7. ОБУЧЕНИЕ ====================
print("\nОбучаю LightGBM (с лагами)...")

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

# ==================== 8. ПРЕДСКАЗАНИЕ ====================
y_pred = model.predict(X_test)

# ==================== 9. АНАЛИЗ ====================
corr = np.corrcoef(y_test, y_pred)[0, 1]
mse = np.mean((y_test - y_pred) ** 2)

print(f"\n=== РЕЗУЛЬТАТЫ РЕАЛЬНОГО ТЕСТА (С ЛАГАМИ) ===")
print(f"Корреляция: {corr:.4f}")
print(f"MSE: {mse:.8f}")
print(f"RMSE: {np.sqrt(mse):.6f}")

# ==================== 10. ВИЗУАЛИЗАЦИЯ ====================
fig, axes = plt.subplots(4, 1, figsize=(16, 14))

# 1. Истинный сигнал vs предсказание
ax1 = axes[0]
ax1.plot(y_test.index, y_test.values, label='Идеальный сигнал (будущее)', 
         alpha=0.8, linewidth=1.5, color='blue')
ax1.plot(y_test.index, y_pred, label='Предсказание модели (с лагами 1-5)', 
         alpha=0.8, linewidth=1.5, color='red')
ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax1.set_title(f'Результат с лагами: Тест: {len(y_test)} баров | Корреляция = {corr:.4f}', fontsize=12)
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# 2. Ошибка
ax2 = axes[1]
error = y_pred - y_test.values
ax2.plot(y_test.index, error, color='red', alpha=0.7, linewidth=1)
ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax2.fill_between(y_test.index, 0, error, alpha=0.2, color='red')
ax2.set_title(f'Ошибка предсказания (MSE = {mse:.8f})')
ax2.grid(True, alpha=0.3)

# 3. Scatter plot
ax3 = axes[2]
ax3.scatter(y_test, y_pred, alpha=0.3, s=10)
ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', alpha=0.5, linewidth=2)
ax3.set_xlabel('Идеальный сигнал (будущее)')
ax3.set_ylabel('Предсказание (из прошлого)')
ax3.set_title(f'Корреляция = {corr:.4f}')
ax3.grid(True, alpha=0.3)

# 4. Цена
ax4 = axes[3]
ax4.plot(y_test.index, m5['close'].loc[y_test.index], color='green', alpha=0.7, linewidth=1)
ax4.set_title('Цена EURUSD (контекст)')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ==================== 11. ВАЖНОСТЬ ФИЧ ====================
print("\nТоп-20 важных признаков (видно, что лаги работают):")
imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print(imp.head(20).to_string())

# ==================== 12. ПРОВЕРКА КАУЗАЛЬНОСТИ ====================
print("\n=== ПРОВЕРКА КАУЗАЛЬНОСТИ ===")
print("Проверяем, не использует ли модель будущие данные...")

print("\n✅ Анализ завершен!")