
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
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
# 1. ЦЕЛЬ (Ground Truth, симметричная свёртка)
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
# 2. ФИЧИ (каузальные + все ТФ)
# ======================================================================
def add_stat_features(df):
    df = df.copy()
    c, h, l, o = df['close'], df['high'], df['low'], df['open']
    v = df['tickvol'] if 'tickvol' in df.columns else (df['tick_volume'] if 'tick_volume' in df.columns else pd.Series(1, index=df.index))

    for period in [20, 50, 100, 200]:
        mean, std = c.rolling(period).mean(), c.rolling(period).std()
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
    sma, std_20 = c.rolling(20).mean(), c.rolling(20).std()
    df['bb_width'] = ((4 * std_20) / (sma + 1e-8)).shift(1)
    df['force_index'] = ((c - c.shift(1)) * v).rolling(20).mean().shift(1)
    return df

print("\nГенерирую каузальные статистические фичи...")
features = add_stat_features(m5)
for tf_name in ['M15', 'M30', 'H1', 'H4', 'DAILY']:
    if tf_name in data:
        tf_df = add_stat_features(data[tf_name])
        tf_resampled = tf_df.resample('5min').ffill()
        for col in tf_resampled.columns:
            if col not in ['open', 'high', 'low', 'close', 'tickvol', 'vol', 'spread', 'tick_volume', 'real_volume']:
                features[f'{tf_name}_{col}'] = tf_resampled[col].shift(1)

features = features.dropna()
print(f"Всего фич: {len(features.columns)}")

common = features.index.intersection(binary.dropna().index)
X = features.loc[common]
y = binary.loc[common].astype(int)
print(f"Всего баров: {len(X)}, фичей: {X.shape[1]}")

# ======================================================================
# 3. WALK-FORWARD + BOOTSTRAP
# ======================================================================
N_SPLITS = 5
TEST_FRAC = 0.15
total = len(X)
test_len = int(total * TEST_FRAC / N_SPLITS)
results = []
train_end = total - test_len * N_SPLITS

print(f"\n=== WALK-FORWARD ({N_SPLITS} окон) ===")
for k in range(N_SPLITS):
    te_start = train_end + k * test_len
    te_end = te_start + test_len if k < N_SPLITS - 1 else total
    X_train, y_train = X.iloc[:te_start], y.iloc[:te_start]
    X_test, y_test = X.iloc[te_start:te_end], y.iloc[te_start:te_end]
    if y_test.nunique() < 2 or y_train.nunique() < 2:
        continue

    model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                               min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
                               reg_alpha=0.5, reg_lambda=0.5, random_state=42, n_jobs=-1, verbose=-1)
    model.fit(X_train, y_train, eval_set=[(X_train, y_train)], eval_metric='binary_logloss',
              callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])

    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)

    B = 1000
    yb, pb = np.array(y_test), np.array(proba)
    idx_boot = np.random.randint(0, len(yb), size=(B, len(yb)))
    boot_auc = np.array([roc_auc_score(yb[idx_boot[b]], pb[idx_boot[b]]) if len(set(yb[idx_boot[b]])) > 1 else np.nan for b in range(B)])
    ci_low, ci_high = np.nanpercentile(boot_auc, [2.5, 97.5])

    results.append({'window': k + 1, 'auc': auc, 'ci_low': ci_low, 'ci_high': ci_high,
                    'test_start': X_test.index[0], 'test_end': X_test.index[-1]})
    print(f"  Окно {k+1}: Test AUC = {auc:.4f}  (95% CI: {ci_low:.4f} – {ci_high:.4f})  "
          f"[{X_test.index[0].strftime('%Y-%m-%d')} → {X_test.index[-1].strftime('%Y-%m-%d')}]")

res_df = pd.DataFrame(results)
print("\n=== ИТОГОВЫЕ МЕТРИКИ ===")
print(f"Средний AUC по окнам: {res_df['auc'].mean():.4f}")
print(f"Мин. AUC:             {res_df['auc'].min():.4f}")
print(f"Макс. AUC:            {res_df['auc'].max():.4f}")
print(f"Ст. отклонение:       {res_df['auc'].std():.4f}")

# ======================================================================
# 4. P&L БЭКТЕСТ
# ======================================================================
SPREAD = 0.00010
SLIPPAGE = 0.00001
COST = SPREAD + 2 * SLIPPAGE
LONG_T = 0.65
SHORT_T = 0.35
MAX_HOLD = 5

print(f"\n=== P&L БЭКТЕСТ (Удержание {MAX_HOLD} баров, без TP/SL) ===")
all_p, all_indices = [], []
for k in range(N_SPLITS):
    te_start = train_end + k * test_len
    te_end = te_start + test_len if k < N_SPLITS - 1 else total
    X_train, y_train = X.iloc[:te_start], y.iloc[:te_start]
    X_test, y_test = X.iloc[te_start:te_end], y.iloc[te_start:te_end]
    if y_test.nunique() < 2:
        continue
    model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                               min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
                               reg_alpha=0.5, reg_lambda=0.5, random_state=42, n_jobs=-1, verbose=-1)
    model.fit(X_train, y_train)
    all_p.append(model.predict_proba(X_test)[:, 1])
    all_indices.extend(X_test.index)

prob_series = pd.Series(np.concatenate(all_p), index=all_indices)
equity, eq_curve, trades, pos, entry, entry_i = 10000.0, [10000.0], [], 0, 0.0, 0

for i, ts in enumerate(prob_series.index):
    prob = prob_series.loc[ts]
    price = m5['close'].loc[ts]
    if pos == 0:
        if prob > LONG_T:
            pos, entry, entry_i = 1, price, i
        elif prob < SHORT_T:
            pos, entry, entry_i = -1, price, i
    else:
        held = i - entry_i
        exit_sig = (pos == 1 and prob < 0.5) or (pos == -1 and prob > 0.5) or (held >= MAX_HOLD)
        if exit_sig:
            gross = (price - entry) if pos == 1 else (entry - price)
            net = gross - COST
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
exp = np.mean(trades) if n_trades else 0.0

print(f"Сделок:      {n_trades}")
print(f"Win rate:    {win_rate:.1f}%")
print(f"Ожидание/сд: {exp*10000:.2f} пунктов")
print(f"Итог P&L:    {total_ret:+.2f}%")
print(f"Max DD:      {mdd:.2f}%")

print("\n✅ Готово. Отключаюсь от MT5.")
mt5.shutdown()