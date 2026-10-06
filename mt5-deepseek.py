import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy import stats
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

# ==================== 1. ПОДКЛЮЧЕНИЕ ====================
print("Подключаюсь к MetaTrader 5...")
if not mt5.initialize():
    print("Ошибка!")
    mt5.shutdown()
    exit()

print(f"Терминал: {mt5.terminal_info()}")
print(f"Авторизован: {mt5.account_info().login}")

# ==================== 2. ПАРАМЕТРЫ ====================
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_M5
BARS_TEST = 500
BARS_TRAIN = 8000
GAUSSIAN_SIGMA = 3.0
LOOKAHEAD = 1

# ==================== 3. ЗАГРУЗКА ====================
print(f"\nЗагружаю {SYMBOL}...")

test_bars = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, BARS_TEST)
test_df = pd.DataFrame(test_bars)
test_df['time'] = pd.to_datetime(test_df['time'], unit='s')
test_df.set_index('time', inplace=True)

train_bars = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, BARS_TEST, BARS_TRAIN)
train_df = pd.DataFrame(train_bars)
train_df['time'] = pd.to_datetime(train_df['time'], unit='s')
train_df.set_index('time', inplace=True)

print(f"Обучающая: {len(train_df)} баров")
print(f"Тестовая: {len(test_df)} баров")

# ==================== 4. ЦЕЛЕВОЙ СИГНАЛ ====================
def calculate_target_improved(df, sigma, lookahead):
    """Улучшенный расчет целевого сигнала"""
    close = df['close'].values
    
    # Сглаживание Гауссом
    smoothed = gaussian_filter1d(close, sigma=sigma, mode='nearest')
    
    # Производная
    derivative = np.gradient(smoothed)
    
    # Нормализация через ATR
    high_low = df['high'].values - df['low'].values
    high_close = np.abs(df['high'].values - df['close'].shift(1).values)
    low_close = np.abs(df['low'].values - df['close'].shift(1).values)
    ranges = np.column_stack([high_low, high_close, low_close])
    true_range = np.max(ranges, axis=1)
    atr = pd.Series(true_range).rolling(14).mean().fillna(0).values
    
    normalized_derivative = derivative / (atr + 1e-8)
    
    # Сдвигаем вперед
    target = np.roll(normalized_derivative, -lookahead)
    target[-lookahead:] = np.nan
    
    # Отсекаем экстремумы
    target = np.clip(target, -0.02, 0.02)
    
    return target, smoothed, derivative, normalized_derivative

train_target, train_smoothed, train_derivative, train_norm_deriv = calculate_target_improved(
    train_df, GAUSSIAN_SIGMA, LOOKAHEAD
)
test_target, test_smoothed, test_derivative, test_norm_deriv = calculate_target_improved(
    test_df, GAUSSIAN_SIGMA, LOOKAHEAD
)

train_df = train_df.iloc[:-LOOKAHEAD].copy()
train_target = train_target[:-LOOKAHEAD]
test_df = test_df.iloc[:-LOOKAHEAD].copy()
test_target = test_target[:-LOOKAHEAD]

print(f"\nЦелевой сигнал (нормированный):")
print(f"  Среднее: {train_target.mean():.6f}")
print(f"  Стандартное отклонение: {train_target.std():.6f}")
print(f"  Диапазон: {train_target.min():.6f} ... {train_target.max():.6f}")

# ==================== 5. СУПЕР-РАСШИРЕННЫЕ ФИЧИ (ИСПРАВЛЕННЫЕ) ====================
def add_super_features(df):
    """Максимально расширенный набор индикаторов"""
    df = df.copy()
    
    # 1. Базовые скользящие средние (сначала создаем все)
    sma_dict = {}
    ema_dict = {}
    for period in [3, 5, 8, 13, 21, 34, 55]:
        sma_dict[f'SMA_{period}'] = df['close'].rolling(period).mean()
        ema_dict[f'EMA_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
    
    # Добавляем SMA и EMA в DataFrame
    for key, value in sma_dict.items():
        df[key] = value
    for key, value in ema_dict.items():
        df[key] = value
    
    # 2. RSI
    def rsi(data, period):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    for period in [5, 9, 14, 21]:
        df[f'RSI_{period}'] = rsi(df['close'], period)
    
    # 3. MACD
    for fast, slow, signal in [(12, 26, 9), (8, 21, 5)]:
        exp1 = df['close'].ewm(span=fast, adjust=False).mean()
        exp2 = df['close'].ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        df[f'MACD_{fast}_{slow}'] = macd
        df[f'MACD_signal_{fast}_{slow}'] = macd.ewm(span=signal, adjust=False).mean()
        df[f'MACD_hist_{fast}_{slow}'] = macd - df[f'MACD_signal_{fast}_{slow}']
    
    # 4. Полосы Боллинджера
    for period in [14, 20, 30]:
        sma = df['close'].rolling(period).mean()
        std = df['close'].rolling(period).std()
        for mult in [1.5, 2.0, 2.5]:
            df[f'BB_upper_{period}_{mult}'] = sma + mult * std
            df[f'BB_lower_{period}_{mult}'] = sma - mult * std
            df[f'BB_%B_{period}_{mult}'] = (df['close'] - df[f'BB_lower_{period}_{mult}']) / (df[f'BB_upper_{period}_{mult}'] - df[f'BB_lower_{period}_{mult}'] + 1e-8)
    
    # 5. ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    for period in [7, 14, 21]:
        df[f'ATR_{period}'] = true_range.rolling(period).mean()
    
    # 6. Стохастик
    for k in [5, 9, 14]:
        low_min = df['low'].rolling(k).min()
        high_max = df['high'].rolling(k).max()
        df[f'Stoch_{k}'] = 100 * ((df['close'] - low_min) / (high_max - low_min + 1e-8))
        df[f'Stoch_signal_{k}'] = df[f'Stoch_{k}'].rolling(3).mean()
    
    # 7. ADX
    def adx(df, period=14):
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = np.abs(high - close.shift())
        tr3 = np.abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        dm_plus = pd.Series(np.where((high - high.shift()) > (low.shift() - low), 
                                     np.maximum(high - high.shift(), 0), 0), index=df.index)
        dm_minus = pd.Series(np.where((low.shift() - low) > (high - high.shift()), 
                                      np.maximum(low.shift() - low, 0), 0), index=df.index)
        
        di_plus = 100 * dm_plus.rolling(period).mean() / atr
        di_minus = 100 * dm_minus.rolling(period).mean() / atr
        
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-8)
        return dx.rolling(period).mean()
    
    df['ADX_14'] = adx(df, 14)
    df['ADX_21'] = adx(df, 21)
    
    # 8. CCI
    def cci(df, period=20):
        tp = (df['high'] + df['low'] + df['close']) / 3
        ma = tp.rolling(period).mean()
        md = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        return (tp - ma) / (0.015 * md + 1e-8)
    
    for period in [14, 20, 30]:
        df[f'CCI_{period}'] = cci(df, period)
    
    # 9. Момент и ROC
    for period in [3, 5, 10, 20]:
        df[f'ROC_{period}'] = df['close'].pct_change(period) * 100
        df[f'momentum_{period}'] = df['close'] - df['close'].shift(period)
    
    # 10. Лаги
    for lag in [1, 2, 3, 5, 8, 13]:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
        df[f'return_lag_{lag}'] = df['close'].pct_change(lag)
    
    # 11. Отношения (теперь SMA уже существуют)
    for period in [20, 50, 100]:
        if f'SMA_{period}' in df.columns:
            df[f'price_to_SMA_{period}'] = df['close'] / df[f'SMA_{period}'] - 1
        df[f'price_to_High_{period}'] = df['close'] / df['high'].rolling(period).max() - 1
        df[f'price_to_Low_{period}'] = df['close'] / df['low'].rolling(period).min() - 1
    
    # 12. Волатильность
    for period in [5, 10, 20, 30]:
        df[f'volatility_{period}'] = df['close'].pct_change().rolling(period).std()
    
    # 13. Свечные характеристики
    df['body'] = np.abs(df['close'] - df['open'])
    df['upper_shadow'] = df['high'] - np.maximum(df['close'], df['open'])
    df['lower_shadow'] = np.minimum(df['close'], df['open']) - df['low']
    df['body_ratio'] = df['body'] / (df['high'] - df['low'] + 1e-8)
    df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
    df['close_open_ratio'] = (df['close'] - df['open']) / df['open']
    
    return df

print("\nГенерирую супер-расширенные индикаторы...")
train_feat = add_super_features(train_df)
test_feat = add_super_features(test_df)

# Удаляем NaN
train_feat = train_feat.dropna()
test_feat = test_feat.dropna()

# Синхронизируем с целевой переменной
train_target = train_target[-len(train_feat):]
test_target = test_target[-len(test_feat):]

# Определяем фичи
exclude_cols = ['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
feature_cols = [col for col in train_feat.columns if col not in exclude_cols]

X_train = train_feat[feature_cols].values
y_train = train_target
X_test = test_feat[feature_cols].values
y_test = test_target

print(f"Всего фич: {len(feature_cols)}")
print(f"Обучающая: {len(X_train)} баров")
print(f"Тестовая: {len(X_test)} баров")

# ==================== 6. МАСШТАБИРОВАНИЕ ====================
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==================== 7. ОБУЧЕНИЕ ====================
print("\nОбучаю LightGBM...")

model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=63,
    max_depth=10,
    min_child_samples=10,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=0.01,
    reg_lambda=0.01,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)],
    eval_metric='rmse',
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
)

# ==================== 8. ПРЕДСКАЗАНИЕ ====================
y_pred = model.predict(X_test_scaled)

# Калибровка предсказаний (если std не равен 0)
if y_pred.std() > 1e-8:
    y_pred_calibrated = y_pred * (y_test.std() / y_pred.std()) + (y_test.mean() - y_pred.mean() * (y_test.std() / y_pred.std()))
else:
    y_pred_calibrated = y_pred.copy()
    print("⚠️ Предсказания имеют нулевую дисперсию!")

# ==================== 9. ДИАГНОСТИКА ====================
print("\n=== ДИАГНОСТИКА ===")
print(f"Истина - среднее: {y_test.mean():.8f}, std: {y_test.std():.8f}")
print(f"Предсказание - среднее: {y_pred.mean():.8f}, std: {y_pred.std():.8f}")
print(f"Калиброванное - среднее: {y_pred_calibrated.mean():.8f}, std: {y_pred_calibrated.std():.8f}")
print(f"Корреляция Пирсона: {np.corrcoef(y_test, y_pred)[0,1]:.4f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")

# ==================== 10. ВИЗУАЛИЗАЦИЯ ====================
fig, axes = plt.subplots(4, 1, figsize=(16, 14))

ax1 = axes[0]
ax1.plot(y_test, label='Истинный сигнал', alpha=0.7, linewidth=1.5, color='blue')
ax1.plot(y_pred, label='Предсказание (сырое)', alpha=0.7, linewidth=1.5, color='orange')
if y_pred.std() > 1e-8:
    ax1.plot(y_pred_calibrated, label='Предсказание (калиброванное)', alpha=0.7, linewidth=1.5, color='green')
ax1.set_title(f'Целевой сигнал vs Предсказание (σ={GAUSSIAN_SIGMA}, {len(y_test)} баров)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)

ax2 = axes[1]
error = y_test - y_pred
ax2.plot(error, label='Ошибка', alpha=0.5, linewidth=1, color='red')
ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax2.set_title(f'MSE: {mean_squared_error(y_test, y_pred):.8f}')
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3 = axes[2]
ax3.scatter(y_test, y_pred, alpha=0.3, s=8)
ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', alpha=0.5, linewidth=2)
ax3.set_xlabel('Истина')
ax3.set_ylabel('Предсказание')
ax3.set_title(f'Корреляция: {np.corrcoef(y_test, y_pred)[0,1]:.4f}')
ax3.grid(True, alpha=0.3)

ax4 = axes[3]
ax4.hist(y_test, bins=50, alpha=0.4, label='Истина', color='blue', density=True)
ax4.hist(y_pred, bins=50, alpha=0.4, label='Предсказание', color='orange', density=True)
ax4.set_title('Распределение значений')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ==================== 11. ВАЖНОСТЬ ФИЧ ====================
print("\nТоп-20 важных признаков:")
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False).head(20)
print(feature_importance.to_string(index=False))

mt5.shutdown()
print("\n✅ Готово!")