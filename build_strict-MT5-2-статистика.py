"""
Модель с использованием СТАТИСТИЧЕСКИХ НАКОПИТЕЛЬНЫХ фич.
Фичи берутся со всех таймфреймов (M5, M15, M30, H1, H4, D1).
Логика:
- Цель: Идеальный учитель (производная Гаусса) на ТЕКУЩЕМ баре.
- Фичи: Только закрытые бары (shift(1)), синхронизированные по времени M5.
- Фичи: Гладкие, не шумят, обладают эффектом памяти (Z-score, CUSUM и т.д.).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

# ==================== 1. ПОДКЛЮЧЕНИЕ К METATRADER ====================
# import MetaTrader5 as mt5
# print("Подключаюсь к MetaTrader 5...")
# if not mt5.initialize():
#     print("Ошибка подключения!")
#     mt5.shutdown()
#     exit()

# ==================== 2. ЗАГРУЗКА ДАННЫХ ====================
def load_mt5_data(symbol, timeframe, bars=10000):
    # --- Реальный код для MT5 (раскомментируйте при использовании) ---
    # rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    # if rates is None:
    #     return None
    # df = pd.DataFrame(rates)
    # df['time'] = pd.to_datetime(df['time'], unit='s')
    # df.set_index('time', inplace=True)
    # df.rename(columns={'tick_volume': 'tickvol', 'real_volume': 'vol'}, inplace=True)
    # return df
    
    # --- Синтетика для демонстрации (замените на реальные данные) ---
    np.random.seed(42)
    n = 8000
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
    'M5': 'M5',
    'M15': 'M15',
    'M30': 'M30',
    'H1': 'H1',
    'H4': 'H4',
    'DAILY': 'D1'
}

BARS = 8000
for name, tf in timeframes.items():
    df = load_mt5_data(SYMBOL, tf, BARS)
    if df is not None and len(df) > 0:
        data[name] = df
        print(f"  {name}: {len(df)} баров")

BASE = "M5"
m5 = data[BASE].copy()

# ==================== 3. ИДЕАЛЬНЫЙ УЧИТЕЛЬ ====================
def gaussian_derivative_target(close, sigma=5):
    smoothed = gaussian_filter1d(np.log(close), sigma=sigma, mode='nearest')
    derivative = np.gradient(smoothed)
    return pd.Series(derivative, index=close.index)

print("\nГенерирую идеальный сигнал...")
SIGMA = 5
ideal_target = gaussian_derivative_target(m5['close'], SIGMA)

target_mean = ideal_target.rolling(500).mean()
target_std = ideal_target.rolling(500).std()
target_norm = (ideal_target - target_mean) / target_std

# ==================== 4. СТАТИСТИЧЕСКИЕ ФИЧИ ====================
def add_stat_features(df):
    """
    Создает статистические накопительные фичи.
    Все фичи сдвинуты на 1 бар назад (shift(1)), чтобы использовать только ЗАКРЫТЫЕ бары.
    """
    df = df.copy()
    c = df['close']
    h = df['high']
    l = df['low']
    o = df['open']
    
    # Универсальная проверка колонки с объемом
    if 'tickvol' in df.columns:
        v = df['tickvol']
    elif 'tick_volume' in df.columns:
        v = df['tick_volume']
    elif 'real_volume' in df.columns:
        v = df['real_volume']
    else:
        v = pd.Series(1, index=df.index) # Заглушка, если объема нет

    # 1. Z-Score (отклонение от нормы, накопительный)
    for period in [20, 50, 100, 200]:
        mean = c.rolling(period).mean()
        std = c.rolling(period).std()
        df[f'zscore_{period}'] = ((c - mean) / (std + 1e-8)).shift(1)
    
    # 2. Кумулятивный импульс (CUSUM) - накопительный
    df['cusum_20'] = (c - c.ewm(span=20).mean()).rolling(100).sum().shift(1)
    df['cusum_50'] = (c - c.ewm(span=50).mean()).rolling(100).sum().shift(1)
    
    # 3. Логарифмическая волатильность (гладкая)
    df['log_vol_30'] = np.log(c.pct_change().rolling(30).std() + 1e-8).shift(1)
    df['log_vol_100'] = np.log(c.pct_change().rolling(100).std() + 1e-8).shift(1)
    
    # 4. Расстояние до экстремумов (тренд)
    df['dist_high_50'] = ((c - h.rolling(50).max()) / (c + 1e-8)).shift(1)
    df['dist_low_50'] = ((c - l.rolling(50).min()) / (c + 1e-8)).shift(1)
    
    # 5. Накопительная корреляция цена-объем
    df['corr_vol_50'] = c.rolling(50).corr(v).shift(1)
    df['corr_vol_100'] = c.rolling(100).corr(v).shift(1)
    
    # 6. Производная цены, сглаженная (медленный импульс)
    df['smoothed_mom'] = c.diff().rolling(20).mean().shift(1)
    
    # 7. Отношение диапазона к телу (гладкое)
    df['range_body_ratio'] = ((h - l) / (abs(c - o) + 1e-8)).rolling(20).mean().shift(1)
    
    # 8. Ширина канала Боллинджера (накопительная)
    sma = c.rolling(20).mean()
    std_20 = c.rolling(20).std()
    df['bb_width'] = ((4 * std_20) / (sma + 1e-8)).shift(1)

    # 9. Индекс силы (накопительный) - Force Index
    df['force_index'] = ((c - c.shift(1)) * v).rolling(20).mean().shift(1)

    return df

print("\nГенерирую статистические фичи (только закрытые бары)...")

# Фичи для M5
features = add_stat_features(m5)

# Добавляем фичи с других таймфреймов (также только закрытые бары!)
for tf_name in ['M15', 'M30', 'H1', 'H4', 'DAILY']:
    if tf_name in data:
        tf_df = add_stat_features(data[tf_name])
        # Ресемплируем на M5 (берем последнее ЗАКРЫТОЕ значение старшего ТФ)
        tf_resampled = tf_df.resample('5min').ffill()
        
        # Добавляем с shift(1): используем только те данные, что были ДО открытия текущего M5 бара!
        for col in tf_resampled.columns:
            if col not in ['open', 'high', 'low', 'close', 'tickvol', 'vol', 'spread', 'tick_volume', 'real_volume']:
                features[f'{tf_name}_{col}'] = tf_resampled[col].shift(1)

# Удаляем NaN
features = features.dropna()

print(f"Всего фич (статистических): {len(features.columns)}")

# ==================== 5. ПОДГОТОВКА ДАННЫХ ====================
# Синхронизируем целевую переменную
target_aligned = target_norm.loc[features.index]

# ЦЕЛЬ: Никуда не двигаем! Прогнозируем идеальный сигнал на ТЕКУЩЕМ баре.
y = target_aligned 

# X: Фичи уже сдвинуты на 1 бар назад внутри add_stat_features
X = features

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
print("\nОбучаю LightGBM на статистических фичах...")

model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=15, # Уменьшили глубину, чтобы меньше ловить шум
    min_child_samples=50, # Увеличили, чтобы сгладить
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=0.5,
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

print(f"\n=== РЕЗУЛЬТАТЫ СО СТАТИСТИЧЕСКИМИ ФИЧАМИ ===")
print(f"Корреляция: {corr:.4f}")
print(f"MSE: {mse:.8f}")
print(f"RMSE: {np.sqrt(mse):.6f}")

# ==================== 10. ВИЗУАЛИЗАЦИЯ ====================
fig, axes = plt.subplots(4, 1, figsize=(16, 14))

# 1. Истинный сигнал vs предсказание
ax1 = axes[0]
ax1.plot(y_test.index, y_test.values, label='Идеальный сигнал (производная Гаусса)', 
         alpha=0.8, linewidth=1.5, color='blue')
ax1.plot(y_test.index, y_pred, label='Предсказание модели (стат. фичи)', 
         alpha=0.8, linewidth=1.5, color='red')
ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax1.set_title(f'Результат со статистическими фичами\nТест: {len(y_test)} баров | Корреляция = {corr:.4f}', fontsize=12)
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
ax3.set_xlabel('Идеальный сигнал')
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
print("\nТоп-20 важных признаков:")
imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print(imp.head(20).to_string())

# ==================== 12. ЗАВЕРШЕНИЕ ====================
print("\n✅ Анализ завершен!")