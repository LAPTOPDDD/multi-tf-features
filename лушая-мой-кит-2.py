import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import MetaTrader5 as mt5

np.random.seed(42)

# ======================================================================
# ПАРАМЕТРЫ (здесь управление)
# ======================================================================
SYMBOL = "EURUSD"
BARS = 50000
SIGMA = 5

SPREAD = 0.00015
SLIPPAGE = 0.00002
COST = SPREAD + 2 * SLIPPAGE
TP_PIPS = 100
SL_PIPS = 50
SIG_THRESHOLD = 0.40
MAX_HOLD = 48
USE_TRAIL = False                 # трейлинг ОТКЛЮЧЁН (опция)
TRAIL_TRIGGER = 0.5

TRAIN_BARS = 4000
TEST_BARS = 500

# ======================================================================
# 0. ДАННЫЕ ИЗ MT5
# ======================================================================
print("Подключаюсь к MetaTrader 5...")
if not mt5.initialize():
    print("Ошибка подключения! Запустите MT5 и войдите в аккаунт.")
    mt5.shutdown()
    exit()

timeframes = {'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4, 'DAILY': mt5.TIMEFRAME_D1}
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

h1 = data['H1'].copy()

# ======================================================================
# 1. ЦЕЛЬ (Ground Truth, НЕ ТРОГАЕМ)
# ======================================================================
def gaussian_derivative_target(close, sigma=5):
    log_c = np.log(close.values)
    smoothed = gaussian_filter1d(log_c, sigma=sigma, mode='nearest')
    deriv = np.gradient(smoothed)
    return pd.Series(deriv, index=close.index)

print("\nСтрою цель (Ground Truth) на H1...")
raw = gaussian_derivative_target(h1['close'], SIGMA)
roll_mean = raw.rolling(500).mean()
roll_std = raw.rolling(500).std()
norm = (raw - roll_mean) / roll_std

# ======================================================================
# 2. ВРЕМЕННЫЕ ФИЧИ (sin/cos)
# ======================================================================
def add_time_features(df_in, prefix=''):
    df = df_in.copy()
    idx = df.index
    df[f'{prefix}sin_hour'] = np.sin(2 * np.pi * idx.hour / 24)
    df[f'{prefix}cos_hour'] = np.cos(2 * np.pi * idx.hour / 24)
    df[f'{prefix}sin_dow'] = np.sin(2 * np.pi * idx.dayofweek / 7)
    df[f'{prefix}cos_dow'] = np.cos(2 * np.pi * idx.dayofweek / 7)
    df[f'{prefix}sin_dom'] = np.sin(2 * np.pi * (idx.day - 1) / 31)
    df[f'{prefix}cos_dom'] = np.cos(2 * np.pi * (idx.day - 1) / 31)
    return df

# ======================================================================
# 3. СТАТИСТИЧЕСКИЕ ФИЧИ
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
# 4. СОБИРАЕМ ФИЧИ (H1 + H4 + DAILY, ресемпл на H1)
# ======================================================================
print("\nГенерирую фичи (H1 + H4 + DAILY)...")
features = add_stat_features(h1)
features = add_time_features(features, prefix='h1_')

for tf_name in ['H4', 'DAILY']:
    if tf_name in data:
        tf_df = add_stat_features(data[tf_name])
        tf_df = add_time_features(tf_df, prefix=f'{tf_name.lower()}_')
        tf_resampled = tf_df.resample('1h').ffill()
        for col in tf_resampled.columns:
            if col not in ['open', 'high', 'low', 'close', 'tickvol', 'vol', 'spread', 'tick_volume', 'real_volume']:
                features[f'{tf_name}_{col}'] = tf_resampled[col]

features = features.shift(1).dropna()
print(f"Всего фич: {len(features.columns)}")

common = features.index.intersection(norm.dropna().index)
X = features.loc[common]
y_reg = norm.loc[common]
print(f"Всего баров (H1): {len(X)}, фичей: {X.shape[1]}")

price_full = h1['close'].values

# ======================================================================
# 5. ТОРГОВЛЯ (TP/SL + фильтр режима + сила сигнала + опциональный трейлинг)
# ======================================================================
def run_backtest(pred, idx_test, price, regime_filter):
    equity = 10000.0
    eq_curve = [equity]
    trades = []
    pos = 0; entry = 0.0; entry_bar = 0; entry_time = None
    tp = sl = 0.0; peak_price = 0.0

    for i in range(len(pred)):
        sig = pred[i]
        if pos != 0:
            held = i - entry_bar
            if pos == 1 and price[i] > peak_price: peak_price = price[i]
            if pos == -1 and price[i] < peak_price: peak_price = price[i]
            if USE_TRAIL:
                prof = (price[i]-entry) if pos == 1 else (entry-price[i])
                if prof > SL_PIPS/10000*TRAIL_TRIGGER:
                    if pos == 1: sl = max(sl, peak_price - SL_PIPS/10000 * 0.5)
                    else:       sl = min(sl, peak_price + SL_PIPS/10000 * 0.5)
            hit_tp = (pos == 1 and price[i] >= tp) or (pos == -1 and price[i] <= tp)
            hit_sl = (pos == 1 and price[i] <= sl) or (pos == -1 and price[i] >= sl)
            exit_sig = (pos == 1 and sig < 0) or (pos == -1 and sig > 0)
            timeout = held >= MAX_HOLD
            if hit_tp or hit_sl or (exit_sig and held > 3) or timeout:
                gross = (price[i]-entry) if pos == 1 else (entry-price[i])
                net = gross - COST
                if hit_tp: net = TP_PIPS/10000 - COST
                if hit_sl: net = -SL_PIPS/10000 - COST
                equity *= (1 + net/entry)
                trades.append({'entry': entry_time, 'exit': idx_test[i],
                               'side': 'BUY' if pos == 1 else 'SELL',
                               'holding': held, 'pips': net*10000, 'result_%': net/entry*100})
                pos = 0
        if pos == 0 and abs(sig) > SIG_THRESHOLD and regime_filter[i]:
            pos = 1 if sig > 0 else -1
            entry, entry_bar, entry_time, peak_price = price[i], i, idx_test[i], price[i]
            if pos == 1:
                tp, sl = entry + TP_PIPS/10000, entry - SL_PIPS/10000
            else:
                tp, sl = entry - TP_PIPS/10000, entry + SL_PIPS/10000
        eq_curve.append(equity)

    if pos != 0:
        net = ((price[-1]-entry) if pos == 1 else (entry-price[-1])) - COST
        equity *= (1 + net/entry)
        trades.append({'entry': entry_time, 'exit': idx_test[-1],
                       'side': 'BUY' if pos == 1 else 'SELL',
                       'holding': len(pred)-entry_bar, 'pips': net*10000, 'result_%': net/entry*100})

    trades_df = pd.DataFrame(trades)
    n = len(trades_df)
    wr = (trades_df['pips'] > 0).mean()*100 if n else 0
    ret = (equity/10000 - 1)*100
    peak = np.maximum.accumulate(eq_curve)
    mdd = ((peak - np.array(eq_curve))/peak).max()*100
    corr = np.corrcoef(y_reg.loc[idx_test], pred)[0, 1] if len(pred) else 0
    return trades_df, eq_curve, ret, wr, mdd, corr

# ======================================================================
# 6. ROLLING WALK-FORWARD
# ======================================================================
total = len(X)
start = 0
all_trades, all_eq, all_pred, all_idx = [], [], [], []
window_stats = []
k = 0

print(f"\n=== ROLLING WALK-FORWARD (H1, TP={TP_PIPS}, SL={SL_PIPS}, thr={SIG_THRESHOLD}) ===")
print(f"=== USE_TRAIL={USE_TRAIL}, MAX_HOLD={MAX_HOLD} ===")

while start + TRAIN_BARS + TEST_BARS <= total:
    te_start = start + TRAIN_BARS
    te_end = te_start + TEST_BARS
    X_train, y_train = X.iloc[:te_start], y_reg.iloc[:te_start]
    X_test, y_test = X.iloc[te_start:te_end], y_reg.iloc[te_start:te_end]
    idx_test = list(X_test.index)
    price = price_full[te_start:te_end]

    # ---- ФИЛЬТР РЕЖИМА: H4 + DAILY z-score согласованы ----
    h4_z = X_test['H4_zscore_50'].values
    d_z = X_test['DAILY_zscore_50'].values
    regime_filter = (np.sign(h4_z) == np.sign(d_z)) & (np.abs(h4_z) > 0.2)

    model = lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.03, num_leaves=31,
        min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1, objective='huber',
        random_state=42, n_jobs=-1, verbose=-1
    )
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train), (X_test, y_test)],
              eval_metric='rmse',
              callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])

    pred = model.predict(X_test)
    trades_df, eq_curve, ret, wr, mdd, corr = run_backtest(pred, idx_test, price, regime_filter)
    all_trades.append(trades_df)
    all_eq.extend(eq_curve[1:])
    all_pred.extend(pred)
    all_idx.extend(idx_test)

    window_stats.append({'window': k+1, 'corr': corr, 'trades': len(trades_df),
                         'win_rate': wr, 'result_%': ret, 'mdd_%': mdd,
                         'start': idx_test[0].strftime('%Y-%m-%d'),
                         'end': idx_test[-1].strftime('%Y-%m-%d')})
    print(f"  Окно {k+1}: corr={corr:.3f} | сделок={len(trades_df)} | win={wr:.1f}% | "
          f"P&L={ret:+.2f}% | DD={mdd:.2f}%  [{idx_test[0].strftime('%Y-%m-%d')} → {idx_test[-1].strftime('%Y-%m-%d')}]")
    start += TEST_BARS
    k += 1

wf = pd.DataFrame(window_stats)
print("\n=== ИТОГО WALK-FORWARD ===")
print(wf[['window', 'corr', 'trades', 'win_rate', 'result_%', 'mdd_%']].to_string(index=False))
print(f"\nСредний P&L: {wf['result_%'].mean():+.2f}%")
print(f"Средняя корреляция: {wf['corr'].mean():.3f}")
print(f"Всего сделок: {wf['trades'].sum()} (с фильтром режима)")

# ======================================================================
# 7. ГРАФИКИ (ТОЛЬКО ТЕСТ, выравнивание длин)
# ======================================================================
L = len(all_idx)
def al(x): return np.array(x)[:L]

fig, axes = plt.subplots(4, 1, figsize=(16, 20))

ax = axes[0]
ax.plot(al(all_idx), al(y_reg.loc[all_idx].values), color='steelblue', lw=0.8, label='Цель (производная Гаусса, H1)')
ax.axhline(0, color='gray', lw=0.8)
ax.set_title('1) Цель (Ground Truth) на H1')
ax.legend(); ax.grid(alpha=0.3)

ax = axes[1]
pred_a = al(all_pred)
ax.plot(al(all_idx), pred_a, color='crimson', lw=0.6, label='Выход модели (H1)')
ax.fill_between(al(all_idx), 0, pred_a, where=(pred_a > 0), color='green', alpha=0.2, label='BUY')
ax.fill_between(al(all_idx), 0, pred_a, where=(pred_a < 0), color='red', alpha=0.2, label='SELL')
ax.axhline(0, color='gray', lw=0.8)
ax.set_title(f'2) Выход модели (средняя corr = {wf["corr"].mean():.3f})')
ax.legend(); ax.grid(alpha=0.3)

ax = axes[2]
all_t = pd.concat(all_trades)
price_test = price_full[[X.index.get_loc(ts) for ts in all_idx]]
ax.plot(al(all_idx), al(price_test), color='black', lw=0.5, label='Цена (H1)')
if len(all_t) > 0:
    buys = all_t[all_t['side'] == 'BUY']; sells = all_t[all_t['side'] == 'SELL']
    ax.scatter(buys['entry'], [price_full[X.index.get_loc(ts)] for ts in buys['entry']],
               marker='^', color='green', s=15, label='BUY', zorder=5)
    ax.scatter(sells['entry'], [price_full[X.index.get_loc(ts)] for ts in sells['entry']],
               marker='v', color='red', s=15, label='SELL', zorder=5)
ax.set_title(f'3) Сделки на цене (всего {len(all_t)} сделок, фильтр режима ON)')
ax.legend(); ax.grid(alpha=0.3)

ax = axes[3]
eq_arr = np.array(all_eq[:L])
ax.plot(al(all_idx), eq_arr, color='green', lw=1.3, label='Equity (rolling WF, H1)')
ax.axhline(10000, color='gray', ls=':', alpha=0.7)
tot = (eq_arr[-1]/10000 - 1)*100 if len(eq_arr) else 0
ax.set_title(f'4) Equity (итог {tot:+.2f}%, win {wf["win_rate"].mean():.1f}%, DD {wf["mdd_%"].mean():.2f}%)')
ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('chart_H1_final.png', dpi=120)
print("\n✅ Сохранён: chart_H1_final.png")
plt.show()

imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nТоп-20 фичей:")
print(imp.head(20).to_string())

print("\n✅ Готово. Отключаюсь от MT5.")
mt5.shutdown()