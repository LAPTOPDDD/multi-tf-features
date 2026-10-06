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

target_mean = ideal_target.rolling(500).mean()
target_std = ideal_target.rolling(500).std()
target_norm = (ideal_target - target_mean) / target_std

THRESHOLD = 0.3
binary_target = pd.Series(np.nan, index=target_norm.index)
binary_target[target_norm > THRESHOLD] = 1
binary_target[target_norm < -THRESHOLD] = 0

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

    for period in [20, 50, 100, 200]:
        mean = c.rolling(period).mean()
        std = c.rolling(period).std()
        df[f'zscore_{period}'] = ((c - mean) / (std + 1e-8)).shift(1)
    
    df['cusum_20'] = (c - c.ewm(span=20).mean()).rolling(100).sum().shift(1)
    df['cusum_50'] = (c - c.ewm(span=50).mean()).rolling(100).sum().shift(1)
    
    df['log_vol_30'] = np.log(c.pct_change().rolling(30).std() + 1e-8).shift(1)
    df['log_vol_100'] = np.log(c.pct_change().rolling(100).std() + 1e-8).shift(1)
    
    df['dist_high_50'] = ((c - h.rolling(50).max()) / (c + 1e-8)).shift(1)
    df['dist_low_50'] = ((c - l.rolling(50).min()) / (c + 1e-8)).shift(1)
    
    df['corr_vol_50'] = c.rolling(50).corr(v).shift(1)
    df['corr_vol_100'] = c.rolling(100).corr(v).shift(1)
    
    df['smoothed_mom'] = c.diff().rolling(20).mean().shift(1)
    df['range_body_ratio'] = ((h - l) / (abs(c - o) + 1e-8)).rolling(20).mean().shift(1)
    
    sma = c.rolling(20).mean()
    std_20 = c.rolling(20).std()
    df['bb_width'] = ((4 * std_20) / (sma + 1e-8)).shift(1)
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
valid_idx = features.index.intersection(binary_target.dropna().index)
X = features.loc[valid_idx]
y = binary_target.loc[valid_idx]
y = y.astype(int)

split_idx = int(len(X) * 0.8)
test_size = 500

X_train = X.iloc[:split_idx]
X_test = X.iloc[-test_size:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[-test_size:]

print(f"\nTrain: {len(X_train)} баров")
print(f"Test: {len(X_test)} баров")

# ==================== 5. ОБУЧЕНИЕ ====================
print("\nОбучаю LightGBM Classifier...")

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
y_pred_proba = model.predict_proba(X_test)[:, 1]

# ==================== 7. АНАЛИЗ МЕТРИК ====================
decision_threshold = 0.5
y_pred_class = (y_pred_proba > decision_threshold).astype(int)

acc = accuracy_score(y_test, y_pred_class)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n=== РЕЗУЛЬТАТЫ БИНАРНОЙ КЛАССИФИКАЦИИ ===")
print(f"Accuracy: {acc:.4f}")
print(f"AUC-ROC: {auc:.4f}")

# ==================== 8. ТОРГОВЫЙ БЭКТЕСТ ====================
# Настраиваем параметры торговли
LONG_THRESHOLD = 0.65   # Вход в лонг при вероятности > 0.65
SHORT_THRESHOLD = 0.35  # Вход в шорт при вероятности < 0.35
EXIT_THRESHOLD = 0.50   # Выход при вероятности 0.50
SPREAD_COST = 0.00012   # Комиссия/спред за сделку (примерно 1.2 пипса на M5)

# Получаем цены на тестовом участке
test_prices = m5['close'].loc[X_test.index]
test_opens = m5['open'].loc[X_test.index]

# Создаем DataFrame для торговли
trade_df = pd.DataFrame(index=X_test.index)
trade_df['close'] = test_prices
trade_df['open'] = test_opens
trade_df['prob'] = y_pred_proba

# Инициализируем переменные
position = 0  # 0 = кэш, 1 = лонг, -1 = шорт
entry_price = 0.0
equity = [10000.0]  # Стартовый капитал
trade_log = []      # Лог сделок

for i in range(1, len(trade_df)):
    current_prob = trade_df['prob'].iloc[i]
    current_open = trade_df['open'].iloc[i]
    prev_close = trade_df['close'].iloc[i-1]
    
    # --- ЛОГИКА ВХОДА ---
    if position == 0:
        if current_prob > LONG_THRESHOLD:
            position = 1
            entry_price = current_open
        elif current_prob < SHORT_THRESHOLD:
            position = -1
            entry_price = current_open
    
    # --- ЛОГИКА ВЫХОДА ---
    elif position == 1:
        if current_prob < EXIT_THRESHOLD:
            # Закрываем лонг
            profit = (current_open - entry_price) - SPREAD_COST
            equity.append(equity[-1] * (1 + profit / entry_price))
            trade_log.append({'type': 'LONG', 'entry': entry_price, 'exit': current_open, 'pnl': profit})
            position = 0
        else:
            # Держим лонг, считаем нереализованную прибыль
            unrealized = (current_open - entry_price) - SPREAD_COST
            equity.append(equity[-1] * (1 + unrealized / entry_price))
            
    elif position == -1:
        if current_prob > EXIT_THRESHOLD:
            # Закрываем шорт
            profit = (entry_price - current_open) - SPREAD_COST
            equity.append(equity[-1] * (1 + profit / entry_price))
            trade_log.append({'type': 'SHORT', 'entry': entry_price, 'exit': current_open, 'pnl': profit})
            position = 0
        else:
            # Держим шорт, считаем нереализованную прибыль
            unrealized = (entry_price - current_open) - SPREAD_COST
            equity.append(equity[-1] * (1 + unrealized / entry_price))

# Если позиция осталась открытой в конце, закрываем по последней цене
if position != 0:
    last_price = trade_df['close'].iloc[-1]
    if position == 1:
        profit = (last_price - entry_price) - SPREAD_COST
        trade_log.append({'type': 'LONG', 'entry': entry_price, 'exit': last_price, 'pnl': profit})
    else:
        profit = (entry_price - last_price) - SPREAD_COST
        trade_log.append({'type': 'SHORT', 'entry': entry_price, 'exit': last_price, 'pnl': profit})

# Формируем кривую эквити
equity_curve = pd.Series(equity, index=trade_df.index[:len(equity)])

# Считаем статистику
total_trades = len(trade_log)
wins = len([t for t in trade_log if t['pnl'] > 0])
losses = len([t for t in trade_log if t['pnl'] < 0])
win_rate = wins / total_trades * 100 if total_trades > 0 else 0
total_pnl = sum([t['pnl'] for t in trade_log])
final_equity = equity_curve.iloc[-1]
max_drawdown = ((equity_curve.cummax() - equity_curve) / equity_curve.cummax()).max() * 100

print(f"\n=== ТОРГОВЫЙ БЭКТЕСТ ===")
print(f"Всего сделок: {total_trades}")
print(f"Прибыльных: {wins} | Убыточных: {losses}")
print(f"Процент прибыльных (Win Rate): {win_rate:.2f}%")
print(f"Общая прибыль (в валюте, 1 лот): {total_pnl:.6f}")
print(f"Итоговый капитал: {final_equity:.2f} (старт 10000)")
print(f"Максимальная просадка (Max DD): {max_drawdown:.2f}%")

# ==================== 9. ВИЗУАЛИЗАЦИЯ ====================
fig, axes = plt.subplots(5, 1, figsize=(16, 18))

# 1. Вероятность vs Реальность
ax1 = axes[0]
ax1.step(y_test.index, y_test.values, where='mid', label='Идеальный сигнал (1=рост, 0=падение)', 
         color='blue', linewidth=1.5, alpha=0.8)
ax1.plot(y_test.index, y_pred_proba, label='Вероятность роста (модель)', 
         color='red', linewidth=1.5)
ax1.axhline(LONG_THRESHOLD, color='green', linestyle='--', alpha=0.7, label=f'Лонг > {LONG_THRESHOLD}')
ax1.axhline(SHORT_THRESHOLD, color='orange', linestyle='--', alpha=0.7, label=f'Шорт < {SHORT_THRESHOLD}')
ax1.axhline(EXIT_THRESHOLD, color='gray', linestyle='--', alpha=0.7, label='Выход 0.5')
ax1.set_title(f'Торговые сигналы: AUC = {auc:.4f} | ACC = {acc:.4f}')
ax1.legend(loc='upper right')
ax1.set_ylim(-0.1, 1.1)
ax1.grid(True, alpha=0.3)

# 2. Цена и точки входа/выхода
ax2 = axes[1]
ax2.plot(trade_df.index, trade_df['close'], color='green', alpha=0.7, linewidth=1, label='Цена')
# Отмечаем сделки
for i, trade in enumerate(trade_log):
    if 'entry' in trade:
        entry_idx = trade_df.index[i] # грубо, но для визуализации сойдет
        if trade['type'] == 'LONG':
            ax2.scatter(entry_idx, trade['entry'], color='green', marker='^', s=50)
        elif trade['type'] == 'SHORT':
            ax2.scatter(entry_idx, trade['entry'], color='red', marker='v', s=50)

ax2.set_title('Точки входа в сделки')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

# 3. Эквити
ax3 = axes[2]
ax3.plot(equity_curve.index, equity_curve.values, color='blue', linewidth=1.5, label='Equity')
ax3.axhline(10000, color='gray', linestyle='--', alpha=0.5, label='Старт')
ax3.set_title(f'Кривая эквити (Итог: {final_equity:.2f} | Max DD: {max_drawdown:.2f}%)')
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3)

# 4. Ошибка
ax4 = axes[3]
error = (y_test.values - y_pred_proba)
ax4.plot(y_test.index, error, color='red', alpha=0.7, linewidth=1)
ax4.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax4.fill_between(y_test.index, 0, error, alpha=0.2, color='red')
ax4.set_title('Ошибка предсказания')
ax4.grid(True, alpha=0.3)

# 5. Scatter
ax5 = axes[4]
ax5.scatter(y_test, y_pred_proba, alpha=0.3, s=10, color='red')
ax5.set_xlabel('Идеальный сигнал (0 или 1)')
ax5.set_ylabel('Предсказанная вероятность (0-1)')
ax5.set_title(f'Scatter: AUC = {auc:.4f}')
ax5.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ==================== 10. ВАЖНОСТЬ ФИЧ ====================
print("\nТоп-20 важных признаков:")
imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print(imp.head(20).to_string())