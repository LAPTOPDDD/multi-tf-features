import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import matplotlib
import matplotlib.pyplot as plt
import MetaTrader5 as mt5

np.random.seed(42)

# ======================================================================
# 0. ДАННЫЕ ИЗ METATRADER 5
# ======================================================================
print("Подключаюсь к MetaTrader 5...")
if not mt5.initialize():
    print("Ошибка подключения! Запустите терминал MT5 и войдите в аккаунт.")
    mt5.shutdown()
    exit()

SYMBOL = "EURUSD"
timeframes = {
    'M5': mt5.TIMEFRAME_M5, 'M15': mt5.TIMEFRAME_M15, 'M30': mt5.TIMEFRAME_M30,
    'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4, 'DAILY': mt5.TIMEFRAME_D1,
}
BARS = 50000
data = {}
for name, tf in timeframes.items():
    rates = mt5.copy_rates_from_pos(SYMBOL, tf, 0, BARS)
    if rates is None:
        continue
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df.rename(columns={'tick_volume': 'tickvol', 'real_volume': 'vol'}, inplace=True)
    data[name] = df
    print(f"  {name}: {len(df)} баров")

m5 = data['M5'].copy()

# ======================================================================
# 1. ЦЕЛЬ (Ground Truth, симметричная свёртка — НЕ ТРОГАЕМ)
# ======================================================================
def gaussian_derivative_target(close, sigma=5):
    log_c = np.log(close.values)
    smoothed = gaussian_filter1d(log_c, sigma=sigma, mode='nearest')
    deriv = np.gradient(smoothed)
    return pd.Series(deriv, index=close.index)

print("\nСтрою цель (Ground Truth)...")
SIGMA = 5
THRESHOLD = 0.3
raw = gaussian_derivative_target(m5['close'], SIGMA)
roll_mean = raw.rolling(500).mean()
roll_std = raw.rolling(500).std()
norm = (raw - roll_mean) / roll_std
binary = pd.Series(np.nan, index=norm.index)
binary[norm > THRESHOLD] = 1
binary[norm < -THRESHOLD] = 0

# ======================================================================
# 2. ВРЕМЕННЫЕ ФИЧИ (циклические sin/cos)
# ======================================================================
def add_time_features(df_in, prefix=''):
    df = df_in.copy()
    idx = df.index
    minute = idx.minute
    df[f'{prefix}sin_minute'] = np.sin(2 * np.pi * minute / 60)
    df[f'{prefix}cos_minute'] = np.cos(2 * np.pi * minute / 60)
    hour = idx.hour
    df[f'{prefix}sin_hour'] = np.sin(2 * np.pi * hour / 24)
    df[f'{prefix}cos_hour'] = np.cos(2 * np.pi * hour / 24)
    dow = idx.dayofweek
    df[f'{prefix}sin_dow'] = np.sin(2 * np.pi * dow / 7)
    df[f'{prefix}cos_dow'] = np.cos(2 * np.pi * dow / 7)
    dom = idx.day
    df[f'{prefix}sin_dom'] = np.sin(2 * np.pi * (dom - 1) / 31)
    df[f'{prefix}cos_dom'] = np.cos(2 * np.pi * (dom - 1) / 31)
    return df

# ======================================================================
# 3. СТАТИСТИЧЕСКИЕ ФИЧИ (rolling, БЕЗ .shift — он будет один раз в конце)
# ======================================================================
def add_stat_features(df):
    df = df.copy()
    c, h, l, o = df['close'], df['high'], df['low'], df['open']
    v = df['tickvol'] if 'tickvol' in df.columns else (df['tick_volume'] if 'tick_volume' in df.columns else pd.Series(1, index=df.index))

    for period in [20, 50, 100, 200]:
        mean, std = c.rolling(period).mean(), c.rolling(period).std()
        df[f'zscore_{period}'] = (c - mean) / (std + 1e-8)

    df['cusum_20'] = (c - c.ewm(span=20).mean()).rolling(100).sum()
    df['smoothed_mom'] = c.diff().rolling(20).mean()
    df['log_vol_30'] = np.log(c.pct_change().rolling(30).std() + 1e-8)
    df['log_vol_100'] = np.log(c.pct_change().rolling(100).std() + 1e-8)
    df['dist_high_50'] = (c - h.rolling(50).max()) / (c + 1e-8)
    df['dist_low_50'] = (c - l.rolling(50).min()) / (c + 1e-8)
    df['corr_vol_50'] = c.rolling(50).corr(v)
    df['corr_vol_100'] = c.rolling(100).corr(v)
    df['range_body_ratio'] = ((h - l) / (abs(c - o) + 1e-8)).rolling(20).mean()
    sma = c.rolling(20).mean()
    std_20 = c.rolling(20).std()
    df['bb_width'] = (4 * std_20) / (sma + 1e-8)
    df['force_index'] = ((c - c.shift(1)) * v).rolling(20).mean()

    return df.iloc[250:]

# ======================================================================
# 4. СОБИРАЕМ ВСЕ ФИЧИ (все ТФ + временные)
# ======================================================================
print("\nГенерирую фичи (все ТФ + время)...")
features = add_stat_features(m5)
features = add_time_features(features, prefix='m5_')

for tf_name in ['M15', 'M30', 'H1', 'H4', 'DAILY']:
    if tf_name in data:
        tf_df = add_stat_features(data[tf_name])
        tf_df = add_time_features(tf_df, prefix=f'{tf_name.lower()}_')
        tf_resampled = tf_df.resample('5min').ffill()
        for col in tf_resampled.columns:
            if col not in ['open', 'high', 'low', 'close', 'tickvol', 'vol', 'spread', 'tick_volume', 'real_volume']:
                features[f'{tf_name}_{col}'] = tf_resampled[col]

# ===== ГЛАВНЫЙ ФИКС: ВСЕ фичи (включая время) -> бар T-1 =====
features = features.shift(1).dropna()
print(f"Всего фич: {len(features.columns)}")

common = features.index.intersection(norm.dropna().index)
X = features.loc[common]
y_reg = norm.loc[common]           # регрессия: предсказываем САМУ производную
print(f"Всего баров: {len(X)}, фичей: {X.shape[1]}")

# ======================================================================
# 5. ОДНО ТЕСТОВОЕ ОКНО (последние 15%)
# ======================================================================
TEST_FRAC = 0.15
split = int(len(X) * (1 - TEST_FRAC))
X_train, y_train = X.iloc[:split], y_reg.iloc[:split]
X_test, y_test = X.iloc[split:], y_reg.iloc[split:]
print(f"\n=== ОДНО ТЕСТОВОЕ ОКНО ===")
print(f"Train: {len(X_train)} | Test (слепой): {len(X_test)}")

model = lgb.LGBMRegressor(
    n_estimators=500, learning_rate=0.03, num_leaves=31,
    min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, objective='huber',
    random_state=42, n_jobs=-1, verbose=-1
)
model.fit(X_train, y_train,
          eval_set=[(X_train, y_train), (X_test, y_test)],
          eval_metric='rmse',
          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

pred = model.predict(X_test)
pred_series = pd.Series(pred, index=X_test.index)
true_series = y_test

# ---- Кросс-корреляция (lag) ----
lags = range(-15, 16)
truth_v = true_series.values - true_series.mean()
pred_v = pred - pred.mean()
corrs = []
for lag in lags:
    if lag < 0:
        corrs.append(np.corrcoef(truth_v[-lag:], pred_v[:len(truth_v)+lag])[0, 1])
    elif lag > 0:
        corrs.append(np.corrcoef(truth_v[:len(truth_v)-lag], pred_v[lag:])[0, 1])
    else:
        corrs.append(np.corrcoef(truth_v, pred_v)[0, 1])
best_lag = lags[int(np.argmax(corrs))]

pred_aligned = pred_series.shift(-best_lag)
mask = pred_aligned.notna() & true_series.notna()
corr_aligned = np.corrcoef(true_series.loc[mask], pred_aligned.loc[mask])[0, 1]
corr_raw = np.corrcoef(true_series, pred_series)[0, 1]

print(f"\nКорреляция (без выравнивания): {corr_raw:.4f}")
print(f"Оптимальный lag: {best_lag} бар(ов)")
print(f"Корреляция (после сдвига на {best_lag}): {corr_aligned:.4f}")

# ======================================================================
# 6. P&L БЭКТЕСТ (сила сигнала + TP/SL)
# ======================================================================
SPREAD = 0.00015
SLIPPAGE = 0.00002
COST = SPREAD + 2 * SLIPPAGE
TP = 0.00020
SL = 0.00012
SIG_STR = 0.25
MAX_HOLD = 30

print(f"\n=== P&L БЭКТЕСТ ===")
equity, eq_curve, trades, pos, entry, entry_i = 10000.0, [10000.0], [], 0, 0.0, 0
price_series = m5['close'].loc[X_test.index]

for i, ts in enumerate(pred_series.index):
    sig = pred_series.loc[ts]
    price = price_series.loc[ts]
    if pos == 0:
        if sig > SIG_STR:
            pos, entry, entry_i = 1, price, i
        elif sig < -SIG_STR:
            pos, entry, entry_i = -1, price, i
    else:
        held = i - entry_i
        if pos == 1:
            if price >= entry + TP:      net = TP - COST
            elif price <= entry - SL:    net = -SL - COST
            elif held >= MAX_HOLD:       net = (price - entry) - COST
            else:                        net = None
        else:
            if price <= entry - TP:      net = TP - COST
            elif price >= entry + SL:    net = -SL - COST
            elif held >= MAX_HOLD:       net = (entry - price) - COST
            else:                        net = None
        if net is not None:
            equity *= (1 + net / entry)
            trades.append(net)
            pos = 0
    eq_curve.append(equity)

trades = np.array(trades)
n_trades = len(trades)
win_rate = (trades > 0).mean() * 100 if n_trades else 0
total_ret = (equity / 10000 - 1) * 100
peak = np.maximum.accumulate(eq_curve)
mdd = ((peak - eq_curve) / peak).max() * 100
print(f"Сделок: {n_trades} | Win: {win_rate:.1f}% | Итог: {total_ret:+.2f}% | DD: {mdd:.2f}%")

# ======================================================================
# 7. ВСЕ ГРАФИКИ
# ======================================================================
fig, axes = plt.subplots(5, 1, figsize=(16, 22), sharex=True)

ax = axes[0]
ax.plot(true_series.index, true_series.values, color='steelblue', lw=1, label='Идеальный сигнал (производная)')
ax.axhline(THRESHOLD, color='green', ls='--', alpha=0.6)
ax.axhline(-THRESHOLD, color='red', ls='--', alpha=0.6)
ax.set_title('1) Целевой сигнал (Ground Truth)')
ax.legend(); ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(pred_series.index, pred_series.values, color='crimson', lw=0.8, label='Выход модели')
ax.set_title(f'2) Выход модели (corr = {corr_raw:.3f})')
ax.legend(); ax.grid(alpha=0.3)

ax = axes[2]
ax.plot(true_series.index, true_series.values, color='steelblue', lw=1, label='Истина')
ax.plot(pred_aligned.index[mask], pred_aligned.loc[mask], color='crimson', lw=0.8, alpha=0.9, label=f'Модель (lag {best_lag})')
ax.set_title(f'3) Наложение (corr = {corr_aligned:.3f})')
ax.legend(); ax.grid(alpha=0.3)

ax = axes[3]
ax.bar(list(lags), corrs, color='purple', width=0.8)
ax.axvline(0, color='black', lw=0.8)
ax.axvline(best_lag, color='red', ls='--', label=f'best lag = {best_lag}')
ax.set_title('4) Кросс-корреляция: lag vs корреляция')
ax.set_xlabel('lag (баров)')
ax.legend(); ax.grid(alpha=0.3)

n = min(len(pred_series.index), len(eq_curve))
ax = axes[4]
ax.plot(pred_series.index[:n], eq_curve[:n], color='green', lw=1.5)
ax.set_title(f'5) Equity (итог {total_ret:+.2f}%, win {win_rate:.1f}%, DD {mdd:.2f}%)')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('chart_full.png', dpi=120)
print("\n✅ Сохранён: chart_full.png")
plt.show()

zoom_n = min(400, len(pred_series))
fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(pred_series.index[:zoom_n], true_series.values[:zoom_n], color='steelblue', lw=1, label='Истина')
ax.plot(pred_series.index[:zoom_n], pred_series.values[:zoom_n], color='crimson', lw=0.8, alpha=0.9, label='Модель')
ax.set_title('Зум: первые бары теста')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('chart_zoom.png', dpi=120)
print("✅ Сохранён: chart_zoom.png")
plt.show()

imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nТоп-20 фичей:")
print(imp.head(20).to_string())

print("\n✅ Готово. Отключаюсь от MT5.")
mt5.shutdown()