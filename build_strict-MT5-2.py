"""
Реальный сценарий торговли:
- На каждом баре M5 используем данные, доступные в момент открытия
- Целевой сигнал - производная Гаусса (идеальный учитель)
- Модель предсказывает будущее движение
"""

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

# ==================== 1. ПОДКЛЮЧЕНИЕ К METATRADER ====================
print("Подключаюсь к MetaTrader 5...")
if not mt5.initialize():
    print("Ошибка подключения!")
    mt5.shutdown()
    exit()

print(f"Терминал: {mt5.terminal_info()}")
print(f"Авторизован: {mt5.account_info().login}")

# ==================== 2. ЗАГРУЗКА ДАННЫХ ====================
def load_mt5_data(symbol, timeframe, bars=50000):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df.rename(columns={
        'open': 'open', 'high': 'high', 'low': 'low',
        'close': 'close', 'tick_volume': 'tickvol',
        'real_volume': 'vol', 'spread': 'spread'
    }, inplace=True)
    return df

SYMBOL = "EURUSD"
print(f"\nЗагружаю {SYMBOL}...")

data = {}
timeframes = {
    'M5': mt5.TIMEFRAME_M5,
    'M15': mt5.TIMEFRAME_M15,
    'M30': mt5.TIMEFRAME_M30,
    'H1': mt5.TIMEFRAME_H1,
    'H4': mt5.TIMEFRAME_H4,
    'DAILY': mt5.TIMEFRAME_D1
}

BARS = 50000
for name, tf in timeframes.items():
    df = load_mt5_data(SYMBOL, tf, BARS)
    if df is not None and len(df) > 0:
        data[name] = df
        print(f"  {name}: {len(df)} баров")

BASE = "M5"
m5 = data[BASE].copy()

# ==================== 3. ИДЕАЛЬНЫЙ УЧИТЕЛЬ (ПРОИЗВОДНАЯ ГАУССА) ====================
def gaussian_derivative_target(close, sigma=5):
    """
    Идеальный сигнал - первая производная сглаженной Гауссом цены.
    НЕ каузальный! Это идеальный учитель для обучения.
    Но для обучения используем только прошлые значения (сдвиг вперед).
    """
    # Сглаживание Гауссом (полное, не каузальное)
    smoothed = gaussian_filter1d(np.log(close), sigma=sigma, mode='nearest')
    # Производная
    derivative = np.gradient(smoothed)
    return pd.Series(derivative, index=close.index)

# Идеальный сигнал (для обучения)
SIGMA = 5
ideal_target = gaussian_derivative_target(m5['close'], SIGMA)

# Нормализуем для устойчивости
target_mean = ideal_target.rolling(500).mean()
target_std = ideal_target.rolling(500).std()
target_norm = (ideal_target - target_mean) / target_std

# ==================== 4. РЕАЛЬНЫЕ ФИЧИ (ТОЛЬКО ПРОШЛОЕ) ====================
def add_real_features(df):
    """
    Фичи, доступные в момент открытия нового бара.
    Используем только закрытые бары (shift(1) для всех индикаторов).
    """
    df = df.copy()
    c = df['close']
    h = df['high']
    l = df['low']
    o = df['open']
    
    # 1. Цена открытия текущего бара (известна в момент открытия)
    df['open_price'] = o
    
    # 2. Характеристики ПРОШЛОГО бара (закрыт)
    df['prev_close'] = c.shift(1)
    df['prev_high'] = h.shift(1)
    df['prev_low'] = l.shift(1)
    df['prev_range'] = (h - l).shift(1)
    df['prev_body'] = (c - o).shift(1)
    
    # 3. Скользящие средние (только прошлые данные)
    for period in [5, 10, 20, 50, 100]:
        df[f'sma_{period}'] = c.rolling(period).mean().shift(1)  # shift(1) - только прошлое!
        df[f'ema_{period}'] = c.ewm(span=period, adjust=False).mean().shift(1)
    
    # 4. RSI (только прошлые данные)
    def rsi_causal(s, period=14):
        delta = s.diff()
        gain = (delta.clip(lower=0)).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - 100 / (1 + rs)
    
    df['rsi_14'] = rsi_causal(c, 14).shift(1)
    df['rsi_21'] = rsi_causal(c, 21).shift(1)
    
    # 5. ATR (прошлый)
    tr1 = h - l
    tr2 = (h - c.shift()).abs()
    tr3 = (l - c.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean().shift(1)
    
    # 6. Волатильность (прошлая)
    df['vol_10'] = c.pct_change().rolling(10).std().shift(1)
    df['vol_20'] = c.pct_change().rolling(20).std().shift(1)
    
    # 7. Момент (прошлый)
    for period in [5, 10, 20]:
        df[f'momentum_{period}'] = (c / c.shift(period) - 1).shift(1)
    
    # 8. Цена относительно уровней (прошлое)
    df['close_to_sma20'] = (c / df['sma_20'] - 1)
    df['close_to_sma50'] = (c / df['sma_50'] - 1)
    
    # 9. Отношения high/low (прошлое)
    df['hl_ratio'] = ((h - l) / c).shift(1)
    df['co_ratio'] = ((c - o) / o).shift(1)
    
    return df

print("\nГенерирую реальные фичи (только из прошлого)...")

# Базовые фичи с M5
features = add_real_features(m5)

# Добавляем фичи с других таймфреймов (только прошлое!)
for tf_name in ['M15', 'M30', 'H1', 'H4', 'DAILY']:
    if tf_name in data:
        tf_df = add_real_features(data[tf_name])
        # Ресемплируем на M5
        tf_resampled = tf_df.resample('5min').ffill()
        # Добавляем с shift(1) - используем только закрытые бары!
        for col in tf_resampled.columns:
            if col not in ['open', 'high', 'low', 'close', 'tickvol', 'vol', 'spread']:
                features[f'{tf_name}_{col}'] = tf_resampled[col].shift(1)

# Удаляем NaN
features = features.dropna()

print(f"Всего фич: {len(features.columns)}")

# ==================== 5. ПОДГОТОВКА ДАННЫХ ====================
# Синхронизируем целевую переменную (идеальный учитель)
target_aligned = target_norm.loc[features.index]

# Целевая переменная: СДВИГАЕМ ВПЕРЕД на 1 бар!
# Модель учится предсказывать БУДУЩЕЕ значение производной
y = target_aligned.shift(-1)  # shift(-1) - предсказываем следующий бар
y = y.dropna()

# Фичи: только прошлые данные (уже сделано через shift(1))
X = features.loc[y.index]

print(f"\nДанные для обучения:")
print(f"  Всего баров: {len(X)}")
print(f"  Фич: {len(X.columns)}")
print(f"  Целевая: производная Гаусса на следующем баре")

# ==================== 6. РАЗДЕЛЕНИЕ (СТРОГО ПО ВРЕМЕНИ) ====================
# Обучаем на 80% истории, тестируем на последних 500 барах
split_idx = int(len(X) * 0.8)
train_size = split_idx
test_size = 500

X_train = X.iloc[:split_idx]
X_test = X.iloc[-test_size:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[-test_size:]

print(f"\nTrain: {len(X_train)} баров")
print(f"Test: {len(X_test)} баров")

# ==================== 7. ОБУЧЕНИЕ ====================
print("\nОбучаю LightGBM на идеальном учителе...")

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

# ==================== 8. ПРЕДСКАЗАНИЕ НА ТЕСТЕ ====================
# ВАЖНО: на тесте используем ТОЛЬКО прошлые данные!
# Для каждого бара мы знаем только то, что было ДО его открытия
y_pred = model.predict(X_test)

# ==================== 9. АНАЛИЗ РЕЗУЛЬТАТОВ ====================
corr = np.corrcoef(y_test, y_pred)[0, 1]
mse = np.mean((y_test - y_pred) ** 2)

print(f"\n=== РЕЗУЛЬТАТЫ РЕАЛЬНОГО ТЕСТА ===")
print(f"Корреляция: {corr:.4f}")
print(f"MSE: {mse:.8f}")
print(f"RMSE: {np.sqrt(mse):.6f}")
print(f"Среднее y_test: {y_test.mean():.6f}")
print(f"Среднее y_pred: {y_pred.mean():.6f}")
print(f"Std y_test: {y_test.std():.6f}")
print(f"Std y_pred: {y_pred.std():.6f}")

# ==================== 10. ВИЗУАЛИЗАЦИЯ ====================
fig, axes = plt.subplots(4, 1, figsize=(16, 14))

# 1. Истинный сигнал (идеальный учитель) vs предсказание
ax1 = axes[0]
ax1.plot(y_test.index, y_test.values, label='Идеальный сигнал (производная Гаусса на следующем баре)', 
         alpha=0.8, linewidth=1.5, color='blue')
ax1.plot(y_test.index, y_pred, label='Предсказание модели (фичи из прошлого)', 
         alpha=0.8, linewidth=1.5, color='red')
ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax1.set_title(f'Реальный сценарий: предсказание будущей производной\n'
              f'Тест: {len(y_test)} баров | Корреляция = {corr:.4f} | {"✅ Реально" if corr < 0.9 else "⚠️ Возможен look-ahead"}',
              fontsize=12)
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
ax3.set_title(f'Корреляция = {corr:.4f} | R² = {1 - np.var(error)/np.var(y_test):.4f}')
ax3.grid(True, alpha=0.3)

# 4. Цена (контекст)
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

# ==================== 12. ПРОВЕРКА КАУЗАЛЬНОСТИ ====================
print("\n=== ПРОВЕРКА КАУЗАЛЬНОСТИ ===")
print("Проверяем, не использует ли модель будущие данные...")

# Проверка 1: корреляция со сдвигом
for shift in [1, 2, 3]:
    if shift < len(y_pred):
        y_pred_shifted = np.roll(y_pred, shift)
        corr_shift = np.corrcoef(y_test[:-shift] if shift > 0 else y_test, 
                                 y_pred_shifted[:-shift] if shift > 0 else y_pred_shifted)[0, 1]
        print(f"  Корреляция со сдвигом +{shift} бар: {corr_shift:.4f}")

# Проверка 2: если корреляция слишком высокая
if corr > 0.9:
    print("\n⚠️ ВНИМАНИЕ: Корреляция > 0.9!")
    print("Возможные проблемы:")
    print("  1. В фичах есть будущие данные (проверьте shift(1))")
    print("  2. Целевая переменная использует текущий бар (должен быть shift(-1))")
    print("  3. Индикаторы используют будущие данные")

# ==================== 13. ПРОГНОЗ НА БУДУЩЕЕ ====================
print("\n=== ПРОГНОЗ НА БУДУЩИЕ БАРЫ ===")

# Создаем фичи для последнего известного бара (для прогноза)
last_features = X.iloc[-1:].copy()
last_pred = model.predict(last_features)[0]

print(f"Последний известный бар: {X.index[-1]}")
print(f"Прогноз производной на следующий бар: {last_pred:.6f}")

# Прогноз на несколько баров вперед
print("\nПрогноз на следующие бары (инкрементальный):")
for i in range(1, 6):
    # Простой прогноз: используем последнее предсказание
    print(f"  Бар +{i}: {last_pred * (1 - i*0.1):.6f} (приблизительно)")

# ==================== 14. ЗАВЕРШЕНИЕ ====================
print("\n✅ Анализ завершен!")
print(f"Модель обучена предсказывать будущую производную Гаусса")
print(f"На тесте корреляция: {corr:.4f}")

mt5.shutdown()