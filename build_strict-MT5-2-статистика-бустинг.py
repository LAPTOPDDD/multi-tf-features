import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

# ==================== 1. ЗАГРУЗКА ДАННЫХ ====================
# ЗАМЕНИТЕ ЭТОТ БЛОК НА ВАШУ ФУНКЦИЮ load_mt5_data
def load_mt5_data(symbol, timeframe, bars=10000):
    # --- Реальный код для MT5 (раскомментируйте) ---
    # rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    # df = pd.DataFrame(rates)
    # df['time'] = pd.to_datetime(df['time'], unit='s')
    # df.set_index('time', inplace=True)
    # df.rename(columns={'tick_volume': 'tickvol', 'real_volume': 'vol'}, inplace=True)
    # return df
    
    # --- Синтетика для демонстрации ---
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
timeframes = {'M5': 'M5', 'M15': 'M15', 'M30': 'M30', 'H1': 'H1', 'H4': 'H4', 'DAILY': 'D1'}
BARS = 8000

for name, tf in timeframes.items():
    df = load_mt5_data(SYMBOL, tf, BARS)
    if df is not None and len(df) > 0:
        data[name] = df
        print(f"  {name}: {len(df)} баров")

BASE = "M5"
m5 = data[BASE].copy()

# ==================== 2. ИДЕАЛЬНЫЙ УЧИТЕЛЬ ====================
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

# ==================== 3. СТАТИСТИЧЕСКИЕ ФИЧИ ====================
def add_stat_features(df):
    df = df.copy()
    c = df['close']
    h = df['high']
    l = df['low']
    o = df['open']
    if 'tickvol' in df.columns:
        v = df['tickvol']
    elif 'tick_volume' in df.columns:
        v = df['tick_volume']
    else:
        v = pd.Series(1, index=df.index)

    # 1. Z-Score
    for period in [20, 50, 100, 200]:
        mean = c.rolling(period).mean()
        std = c.rolling(period).std()
        df[f'zscore_{period}'] = ((c - mean) / (std + 1e-8)).shift(1)
    
    # 2. CUSUM
    df['cusum_20'] = (c - c.ewm(span=20).mean()).rolling(100).sum().shift(1)
    df['cusum_50'] = (c - c.ewm(span=50).mean()).rolling(100).sum().shift(1)
    
    # 3. Лог-волатильность
    df['log_vol_30'] = np.log(c.pct_change().rolling(30).std() + 1e-8).shift(1)
    df['log_vol_100'] = np.log(c.pct_change().rolling(100).std() + 1e-8).shift(1)
    
    # 4. Расстояние до экстремумов
    df['dist_high_50'] = ((c - h.rolling(50).max()) / (c + 1e-8)).shift(1)
    df['dist_low_50'] = ((c - l.rolling(50).min()) / (c + 1e-8)).shift(1)
    
    # 5. Накопительная корреляция цена-объем
    df['corr_vol_50'] = c.rolling(50).corr(v).shift(1)
    df['corr_vol_100'] = c.rolling(100).corr(v).shift(1)
    
    # 6. Сглаженный импульс
    df['smoothed_mom'] = c.diff().rolling(20).mean().shift(1)
    
    # 7. Отношение диапазона к телу
    df['range_body_ratio'] = ((h - l) / (abs(c - o) + 1e-8)).rolling(20).mean().shift(1)
    
    # 8. Ширина канала Боллинджера
    sma = c.rolling(20).mean()
    std_20 = c.rolling(20).std()
    df['bb_width'] = ((4 * std_20) / (sma + 1e-8)).shift(1)

    # 9. Force Index
    df['force_index'] = ((c - c.shift(1)) * v).rolling(20).mean().shift(1)

    return df

print("\nГенерирую статистические фичи (только закрытые бары)...")
features = add_stat_features(m5)

# Фичи с других таймфреймов
for tf_name in ['M15', 'M30', 'H1', 'H4', 'DAILY']:
    if tf_name in data:
        tf_df = add_stat_features(data[tf_name])
        tf_resampled = tf_df.resample('5min').ffill()
        for col in tf_resampled.columns:
            if col not in ['open', 'high', 'low', 'close', 'tickvol', 'vol', 'spread', 'tick_volume', 'real_volume']:
                features[f'{tf_name}_{col}'] = tf_resampled[col].shift(1)

features = features.dropna()
print(f"Всего фич: {len(features.columns)}")

# ==================== 4. РАЗДЕЛЕНИЕ ====================
target_aligned = target_norm.loc[features.index]
y = target_aligned
X = features

split_idx = int(len(X) * 0.8)
test_size = 500

X_train = X.iloc[:split_idx]
X_test = X.iloc[-test_size:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[-test_size:]

print(f"\nTrain: {len(X_train)} баров")
print(f"Test: {len(X_test)} баров")

# ==================== 5. ОБУЧЕНИЕ МОДЕЛИ №1 ====================
print("\nОбучаю Модель №1 (Базовая)...")
model1 = lgb.LGBMRegressor(
    n_estimators=500, learning_rate=0.05, num_leaves=15,
    min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.5, reg_lambda=0.5, random_state=42, n_jobs=-1, verbose=-1
)
model1.fit(X_train, y_train)

# ==================== 6. ОБУЧЕНИЕ МОДЕЛИ №2 (Stacking) ====================
print("\nСоздаю мета-фичи для Модели №2...")

# Сначала предсказания на тренировке и тесте от Модели 1
pred_train_1 = model1.predict(X_train) # numpy array
pred_test_1 = model1.predict(X_test)   # numpy array

# Создаем DataFrame для мета-фич
train_meta = pd.DataFrame(index=X_train.index)
test_meta = pd.DataFrame(index=X_test.index)

# 1. Сам прогноз Модели 1
train_meta['pred_m1'] = pred_train_1
test_meta['pred_m1'] = pred_test_1

# 2. Лаги ошибки Модели 1 (как она ошиблась в прошлом)
# Считаем ошибку на тренировке
train_err = y_train - pred_train_1
test_err = y_test - pred_test_1

for lag in range(1, 6): # Берём 5 лагов ошибки
    train_meta[f'err_lag{lag}'] = train_err.shift(lag)
    test_meta[f'err_lag{lag}'] = test_err.shift(lag)

# 3. Добавляем небольшой набор самых важных исходных фич для контекста
top_features = ['zscore_100', 'cusum_50', 'dist_low_50'] # Замените на свои топ-фичи, если нужно
for col in top_features:
    if col in X_train.columns:
        train_meta[col] = X_train[col]
        test_meta[col] = X_test[col]

# Убираем NaN (из-за лагов ошибки)
train_meta = train_meta.dropna()
test_meta = test_meta.dropna()

# Выравниваем цели по индексам
y_train_meta = y_train.loc[train_meta.index]
y_test_meta = y_test.loc[test_meta.index]

print(f"Мета-фич: {len(train_meta.columns)}")

# Обучаем Модель №2 (Максимально простая, чтобы не переобучиться на шум)
print("\nОбучаю Модель №2 (Корректор)...")
model2 = lgb.LGBMRegressor(
    n_estimators=300, 
    learning_rate=0.01, # Очень маленький шаг, чтобы корректировать аккуратно
    num_leaves=3,       # Очень простое дерево
    min_child_samples=100,
    reg_lambda=1.0,     # Сильная регуляризация
    random_state=42, 
    n_jobs=-1, 
    verbose=-1
)
model2.fit(train_meta, y_train_meta)

# ==================== 7. ИТОГОВОЕ ПРЕДСКАЗАНИЕ ====================
# Модель 2 сама решает, как скомбинировать прогноз Модели 1 и её ошибки
y_pred_final = model2.predict(test_meta)

# ==================== 8. АНАЛИЗ ====================
# Работаем с индексами напрямую, без .loc
y_test_vals = y_test_meta.values
pred_test_1_vals = pred_test_1[train_meta.shape[0]:] # Предсказание первой модели для тестовых строк (учитывая сдвиг индексов)
# Но так как мы выкинули NaN из test_meta, нужно аккуратно сопоставить.
# Проще взять предсказания из test_meta по индексу, но test_meta построен на основе X_test.

# Самый простой и точный способ: найти в pred_test_1 значения, соответствующие индексам test_meta
# Для этого создадим Series из предсказаний
pred_test_1_series = pd.Series(pred_test_1, index=X_test.index)
pred_test_1_aligned = pred_test_1_series.loc[test_meta.index]

corr_1 = np.corrcoef(y_test_meta, pred_test_1_aligned)[0, 1]
corr_final = np.corrcoef(y_test_meta, y_pred_final)[0, 1]
mse_final = np.mean((y_test_meta - y_pred_final) ** 2)

print(f"\n=== РЕЗУЛЬТАТЫ STACKING (Стекинг) ===")
print(f"Корреляция Модели 1 (базовой): {corr_1:.4f}")
print(f"Корреляция ИТОГОВОЙ (1+2): {corr_final:.4f}")
print(f"MSE Итоговой: {mse_final:.8f}")

# ==================== 9. ВИЗУАЛИЗАЦИЯ ====================
fig, axes = plt.subplots(4, 1, figsize=(16, 14))

# 1. Сигнал vs Предсказания
ax1 = axes[0]
ax1.plot(y_test_meta.index, y_test_meta.values, label='Идеальный сигнал', color='blue', linewidth=1.5)
ax1.plot(y_test_meta.index, pred_test_1_aligned.values, label='Модель 1 (базовая)', color='orange', alpha=0.7, linewidth=1)
ax1.plot(y_test_meta.index, y_pred_final, label='Итог (Stacking)', color='red', linewidth=1.5)
ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax1.set_title(f'Stacking: Корреляция 1-й = {corr_1:.4f} | Итог = {corr_final:.4f}')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# 2. Ошибка
ax2 = axes[1]
error = y_pred_final - y_test_meta.values
ax2.plot(y_test_meta.index, error, color='red', alpha=0.7, linewidth=1)
ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax2.fill_between(y_test_meta.index, 0, error, alpha=0.2, color='red')
ax2.set_title(f'Ошибка Итоговой модели (MSE = {mse_final:.8f})')
ax2.grid(True, alpha=0.3)

# 3. Scatter
ax3 = axes[2]
ax3.scatter(y_test_meta, y_pred_final, alpha=0.3, s=10, color='red')
ax3.plot([y_test_meta.min(), y_test_meta.max()], [y_test_meta.min(), y_test_meta.max()], 'r--', alpha=0.5, linewidth=2)
ax3.set_xlabel('Идеальный сигнал')
ax3.set_ylabel('Итоговое предсказание')
ax3.set_title('Scatter Stacking модели')
ax3.grid(True, alpha=0.3)

# 4. Цена
ax4 = axes[3]
ax4.plot(y_test_meta.index, m5['close'].loc[y_test_meta.index], color='green', alpha=0.7, linewidth=1)
ax4.set_title('Цена (контекст)')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ==================== 10. ВАЖНОСТЬ ФИЧ В МОДЕЛИ 2 ====================
print("\nТоп-10 важных мета-фич в Модели 2 (Корректоре):")
imp = pd.Series(model2.feature_importances_, index=train_meta.columns).sort_values(ascending=False)
print(imp.head(10).to_string())