import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import accuracy_score, roc_auc_score

# ==================== 1. ПОДКЛЮЧЕНИЕ К METATRADER 5 ====================
import MetaTrader5 as mt5

print("Подключаюсь к MetaTrader 5...")
if not mt5.initialize():
    print("Ошибка подключения к MT5!")
    mt5.shutdown()
    exit()

print(f"Терминал: {mt5.terminal_info().name}")
print(f"Авторизован на счете: {mt5.account_info().login}")

# ==================== 2. ЗАГРУЗКА ДАННЫХ ИЗ MT5 ====================
def load_mt5_data(symbol, timeframe, bars=50000):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None or len(rates) == 0:
        print(f"Ошибка: Не удалось загрузить данные")
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df.rename(columns={'tick_volume': 'tickvol', 'real_volume': 'vol'}, inplace=True)
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

# ==================== 3. ИДЕАЛЬНЫЙ УЧИТЕЛЬ (БИНАРНЫЙ) ====================
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

# ==================== 4. СТАТИСТИЧЕСКИЕ ФИЧИ ====================
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

# ==================== 5. РАЗДЕЛЕНИЕ НА TRAIN / VALIDATION / TEST ====================
valid_idx = features.index.intersection(binary_target.dropna().index)
X = features.loc[valid_idx]
y = binary_target.loc[valid_idx]
y = y.astype(int)

# Разделяем строго по времени (TimeSeriesSplit)
total_len = len(X)
train_size = int(total_len * 0.70)  # 70% на обучение
val_size = int(total_len * 0.15)   # 15% на валидацию
test_size = total_len - train_size - val_size  # Последние 15% на слепой тест

X_train = X.iloc[:train_size]
X_val = X.iloc[train_size:train_size + val_size]
X_test = X.iloc[train_size + val_size:]

y_train = y.iloc[:train_size]
y_val = y.iloc[train_size:train_size + val_size]
y_test = y.iloc[train_size + val_size:]

print(f"\n=== РАЗДЕЛЕНИЕ ДАННЫХ ===")
print(f"Train: {len(X_train)} баров (70%)")
print(f"Validation: {len(X_val)} баров (15%)")
print(f"Test (слепой): {len(X_test)} баров (15%)")

# ==================== 6. ОБУЧЕНИЕ (БЕЗ УТЕЧКИ!) ====================
print("\nОбучаю LightGBM на Train, останавливаюсь на Validation...")

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
    eval_set=[(X_train, y_train), (X_val, y_val)],  # Валидация вместо теста!
    eval_metric='binary_logloss',
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
)

# ==================== 7. ПРЕДСКАЗАНИЕ ====================
print("\nПредсказываю на Validation и Test...")
y_pred_proba_val = model.predict_proba(X_val)[:, 1]
y_pred_proba_test = model.predict_proba(X_test)[:, 1]

# ==================== 8. АНАЛИЗ МЕТРИК ====================
acc_val = accuracy_score(y_val, (y_pred_proba_val > 0.5).astype(int))
auc_val = roc_auc_score(y_val, y_pred_proba_val)

acc_test = accuracy_score(y_test, (y_pred_proba_test > 0.5).astype(int))
auc_test = roc_auc_score(y_test, y_pred_proba_test)

print(f"\n=== МЕТРИКИ НА ВАЛИДАЦИИ ===")
print(f"Accuracy: {acc_val:.4f}")
print(f"AUC-ROC: {auc_val:.4f}")

print(f"\n=== МЕТРИКИ НА СЛЕПОМ ТЕСТЕ (Честный результат) ===")
print(f"Accuracy: {acc_test:.4f}")
print(f"AUC-ROC: {auc_test:.4f}")

# ==================== 9. ТОРГОВЫЙ БЭКТЕСТ ====================
# Параметры торговли
LONG_THRESHOLD = 0.85
SHORT_THRESHOLD = 0.15
EXIT_THRESHOLD = 0.50
SPREAD_COST = 0.00012

# Функция для выполнения бэктеста на любом участке данных
def run_backtest(X_data, y_proba, prices_df):
    trade_df = pd.DataFrame(index=X_data.index)
    trade_df['close'] = prices_df['close'].loc[X_data.index]
    trade_df['open'] = prices_df['open'].loc[X_data.index]
    trade_df['prob'] = y_proba

    position = 0
    entry_price = 0.0
    equity = [10000.0]
    trade_log = []

    for i in range(1, len(trade_df)):
        current_prob = trade_df['prob'].iloc[i]
        current_open = trade_df['open'].iloc[i]
        
        if position == 0:
            if current_prob > LONG_THRESHOLD:
                position = 1
                entry_price = current_open
            elif current_prob < SHORT_THRESHOLD:
                position = -1
                entry_price = current_open
        
        elif position == 1:
            if current_prob < EXIT_THRESHOLD:
                profit = (current_open - entry_price) - SPREAD_COST
                equity.append(equity[-1] * (1 + profit / entry_price))
                trade_log.append({'type': 'LONG', 'entry': entry_price, 'exit': current_open, 'pnl': profit})
                position = 0
            else:
                unrealized = (current_open - entry_price) - SPREAD_COST
                equity.append(equity[-1] * (1 + unrealized / entry_price))
                
        elif position == -1:
            if current_prob > EXIT_THRESHOLD:
                profit = (entry_price - current_open) - SPREAD_COST
                equity.append(equity[-1] * (1 + profit / entry_price))
                trade_log.append({'type': 'SHORT', 'entry': entry_price, 'exit': current_open, 'pnl': profit})
                position = 0
            else:
                unrealized = (entry_price - current_open) - SPREAD_COST
                equity.append(equity[-1] * (1 + unrealized / entry_price))

    if position != 0:
        last_price = trade_df['close'].iloc[-1]
        if position == 1:
            profit = (last_price - entry_price) - SPREAD_COST
            trade_log.append({'type': 'LONG', 'entry': entry_price, 'exit': last_price, 'pnl': profit})
        else:
            profit = (entry_price - last_price) - SPREAD_COST
            trade_log.append({'type': 'SHORT', 'entry': entry_price, 'exit': last_price, 'pnl': profit})

    equity_curve = pd.Series(equity, index=trade_df.index[:len(equity)])
    
    total_trades = len(trade_log)
    wins = len([t for t in trade_log if t['pnl'] > 0])
    losses = len([t for t in trade_log if t['pnl'] < 0])
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0
    total_pnl = sum([t['pnl'] for t in trade_log])
    final_equity = equity_curve.iloc[-1]
    max_drawdown = ((equity_curve.cummax() - equity_curve) / equity_curve.cummax()).max() * 100

    return {
        'equity': equity_curve,
        'total_trades': total_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'final_equity': final_equity,
        'max_drawdown': max_drawdown,
        'trade_log': trade_log
    }

print("\n=== ТОРГОВЛЯ НА ВАЛИДАЦИИ ===")
backtest_val = run_backtest(X_val, y_pred_proba_val, m5)
print(f"Сделок: {backtest_val['total_trades']} | Win Rate: {backtest_val['win_rate']:.2f}% | Итог: {backtest_val['final_equity']:.2f}")

print("\n=== ТОРГОВЛЯ НА СЛЕПОМ ТЕСТЕ (Честный результат) ===")
backtest_test = run_backtest(X_test, y_pred_proba_test, m5)
print(f"Сделок: {backtest_test['total_trades']} | Win Rate: {backtest_test['win_rate']:.2f}% | Итог: {backtest_test['final_equity']:.2f}")

# ==================== 10. ВИЗУАЛИЗАЦИЯ ====================
fig, axes = plt.subplots(5, 1, figsize=(16, 20))

# 1. Вероятность и сигналы на Validation
ax1 = axes[0]
ax1.step(y_val.index, y_val.values, where='mid', label='Идеальный сигнал (Val)', color='blue', linewidth=1.5, alpha=0.8)
ax1.plot(y_val.index, y_pred_proba_val, label='Вероятность (Val)', color='red', linewidth=1.5)
ax1.axhline(LONG_THRESHOLD, color='green', linestyle='--', alpha=0.7)
ax1.axhline(SHORT_THRESHOLD, color='orange', linestyle='--', alpha=0.7)
ax1.set_title(f'Validation: AUC = {auc_val:.4f}')
ax1.set_ylim(-0.1, 1.1)
ax1.grid(True, alpha=0.3)

# 2. Эквити на Validation
ax2 = axes[1]
ax2.plot(backtest_val['equity'].index, backtest_val['equity'].values, color='blue', linewidth=1.5)
ax2.set_title(f"Validation Equity (Итог: {backtest_val['final_equity']:.2f} | DD: {backtest_val['max_drawdown']:.2f}%)")
ax2.grid(True, alpha=0.3)

# 3. Вероятность и сигналы на Test (Слепой тест)
ax3 = axes[2]
ax3.step(y_test.index, y_test.values, where='mid', label='Идеальный сигнал (Test)', color='blue', linewidth=1.5, alpha=0.8)
ax3.plot(y_test.index, y_pred_proba_test, label='Вероятность (Test)', color='red', linewidth=1.5)
ax3.axhline(LONG_THRESHOLD, color='green', linestyle='--', alpha=0.7)
ax3.axhline(SHORT_THRESHOLD, color='orange', linestyle='--', alpha=0.7)
ax3.set_title(f'Test (Слепой): AUC = {auc_test:.4f}')
ax3.set_ylim(-0.1, 1.1)
ax3.grid(True, alpha=0.3)

# 4. Эквити на Test (Слепой тест)
ax4 = axes[3]
ax4.plot(backtest_test['equity'].index, backtest_test['equity'].values, color='green', linewidth=1.5)
ax4.set_title(f"Test Equity (Итог: {backtest_test['final_equity']:.2f} | DD: {backtest_test['max_drawdown']:.2f}%)")
ax4.grid(True, alpha=0.3)

# 5. Цена (контекст)
ax5 = axes[4]
ax5.plot(m5['close'].loc[X_test.index], color='green', alpha=0.7, linewidth=1)
ax5.set_title('Цена EURUSD на тесте')
ax5.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ==================== 11. ВАЖНОСТЬ ФИЧ ====================
print("\nТоп-20 важных признаков:")
imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print(imp.head(20).to_string())

# ==================== 12. ЗАВЕРШЕНИЕ ====================
print("\n✅ Анализ завершен! Отключаюсь от MT5.")
mt5.shutdown()