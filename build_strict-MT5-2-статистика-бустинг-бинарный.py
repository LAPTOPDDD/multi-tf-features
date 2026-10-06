import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ==================== 1. ЗАГРУЗКА ДАННЫХ ====================
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

# ==================== 2. ИДЕАЛЬНЫЙ УЧИТЕЛЬ (БИНАРНЫЙ) ====================
def gaussian_derivative_target(close, sigma=5):
    smoothed = gaussian_filter1d(np.log(close), sigma=sigma, mode='nearest')
    derivative = np.gradient(smoothed)
    return pd.Series(derivative, index=close.index)

print("\nГенерирую бинарный целевой сигнал...")
SIGMA = 5
ideal_target = gaussian_derivative_target(m5['close'], SIGMA)

# Нормализуем, чтобы задать порог
target_mean = ideal_target.rolling(500).mean()
target_std = ideal_target.rolling(500).std()
target_norm = (ideal_target - target_mean) / target_std

# Превращаем в бинарный вид
# Если значение > 0.3 -> 1 (рост)
# Если значение < -0.3 -> 0 (падение)
# Если между -0.3 и 0.3 -> исключаем из обучения (флэт, шум)
THRESHOLD = 0.3
binary_target = pd.Series(np.nan, index=target_norm.index)
binary_target[target_norm > THRESHOLD] = 1
binary_target[target_norm < -THRESHOLD] = 0

# Сколько у нас классов
print(f"Класс 1 (рост): {(binary_target == 1).sum()}")
print(f"Класс 0 (падение): {(binary_target == 0).sum()}")
print(f"Исключено (флэт): {binary_target.isna().sum()}")

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

print("\nГенерирую статистические фичи...")
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

# ==================== 4. СИНХРОНИЗАЦИЯ И РАЗДЕЛЕНИЕ ====================
# Находим пересечение индексов (где есть и фичи, и бинарная цель)
valid_idx = features.index.intersection(binary_target.dropna().index)
X = features.loc[valid_idx]
y = binary_target.loc[valid_idx]

# Выравниваем (y должен быть int)
y = y.astype(int)

split_idx = int(len(X) * 0.8)
test_size = 500

X_train = X.iloc[:split_idx]
X_test = X.iloc[-test_size:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[-test_size:]

print(f"\nTrain: {len(X_train)} баров")
print(f"Test: {len(X_test)} баров")

# ==================== 5. ОБУЧЕНИЕ КЛАССИФИКАТОРА ====================
print("\nОбучаю LightGBM Classifier (бинарная классификация)...")

model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=15,
    min_child_samples=50,
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
    eval_metric='binary_logloss',
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
)

# ==================== 6. ПРЕДСКАЗАНИЕ ====================
# Предсказываем вероятность класса 1 (рост)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# ==================== 7. АНАЛИЗ МЕТРИК ====================
# Порог для принятия решения
decision_threshold = 0.5
y_pred_class = (y_pred_proba > decision_threshold).astype(int)

acc = accuracy_score(y_test, y_pred_class)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n=== РЕЗУЛЬТАТЫ БИНАРНОЙ КЛАССИФИКАЦИИ ===")
print(f"Accuracy: {acc:.4f}")
print(f"AUC-ROC: {auc:.4f} (0.5 = случайность, 1.0 = идеал)")

# ==================== 8. ВИЗУАЛИЗАЦИЯ ====================
fig, axes = plt.subplots(4, 1, figsize=(16, 14))

# 1. Вероятность vs Реальность
ax1 = axes[0]
# Рисуем реальный бинарный сигнал (шаги)
ax1.step(y_test.index, y_test.values, where='mid', label='Идеальный сигнал (1=рост, 0=падение)', 
         color='blue', linewidth=1.5, alpha=0.8)
# Рисуем вероятность
ax1.plot(y_test.index, y_pred_proba, label='Вероятность роста (модель)', 
         color='red', linewidth=1.5)
ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Порог 0.5')
ax1.set_title(f'Бинарная классификация: AUC = {auc:.4f} | ACC = {acc:.4f}')
ax1.legend(loc='upper right')
ax1.set_ylim(-0.1, 1.1)
ax1.grid(True, alpha=0.3)

# 2. Ошибка классификации
ax2 = axes[1]
# Если реальный сигнал 1, а вероятность < 0.5, то ошибка
error = (y_test.values - y_pred_proba)
ax2.plot(y_test.index, error, color='red', alpha=0.7, linewidth=1)
ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax2.fill_between(y_test.index, 0, error, alpha=0.2, color='red')
ax2.set_title('Ошибка (разница между реальным бинарным сигналом и вероятностью)')
ax2.grid(True, alpha=0.3)

# 3. Scatter вероятности
ax3 = axes[2]
ax3.scatter(y_test, y_pred_proba, alpha=0.3, s=10, color='red')
ax3.set_xlabel('Идеальный бинарный сигнал (0 или 1)')
ax3.set_ylabel('Предсказанная вероятность (0-1)')
ax3.set_title(f'Scatter: AUC = {auc:.4f}')
ax3.grid(True, alpha=0.3)

# 4. Цена
ax4 = axes[3]
ax4.plot(y_test.index, m5['close'].loc[y_test.index], color='green', alpha=0.7, linewidth=1)
ax4.set_title('Цена (контекст)')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ==================== 9. ВАЖНОСТЬ ФИЧ ====================
print("\nТоп-20 важных признаков (классификатор):")
imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print(imp.head(20).to_string())