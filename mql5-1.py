import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import MetaTrader5 as mt5

np.random.seed(42)

# ======================================================================
# ПАРАМЕТРЫ
# ======================================================================
SYMBOL = "EURUSD"
BASE_TF = mt5.TIMEFRAME_H1
HTF = mt5.TIMEFRAME_H4
DAILY_TF = mt5.TIMEFRAME_D1
BARS = 50000

ATR_PERIOD = 10
ATR_MULTIPLIER = 3.0
USE_HTF_FILTER = True

LOT_SIZE = 0.01
INITIAL_BALANCE = 10000.0
TP_PERCENT = 1.0
MAX_RISK_PERCENT = 15.0
MAX_POSITIONS_PER_SIDE = 25

SIGMA = 5
LGBM_THRESHOLD = 0.0

TRAIN_BARS = 4000
TEST_BARS = 500

# ======================================================================
# 0. ДАННЫЕ ИЗ MT5
# ======================================================================
print("Подключаюсь к MetaTrader 5...")
if not mt5.initialize():
    print("Ошибка! Запустите MT5 и войдите в аккаунт.")
    mt5.shutdown()
    exit()

def load(tf, bars):
    rates = mt5.copy_rates_from_pos(SYMBOL, tf, 0, bars)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df[['open', 'high', 'low', 'close']]

print("Качаю реальные данные из MT5...")
base = load(BASE_TF, BARS)
htf_df = load(HTF, BARS)
daily_df = load(DAILY_TF, BARS)
print(f"  H1: {len(base)} | H4: {len(htf_df)} | DAILY: {len(daily_df)}")

price_full = base['close'].values.copy()
idx_base = base.index   # ОПОРНЫЙ индекс (50000 меток)

# ======================================================================
# 1. ATR (Wilder) — numpy
# ======================================================================
def atr(high, low, close, period=10):
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_vals = np.full(n, np.nan)
    if n >= period:
        atr_vals[period-1] = np.mean(tr[:period])
        for i in range(period, n):
            atr_vals[i] = (atr_vals[i-1] * (period - 1) + tr[i]) / period
    return atr_vals

# ======================================================================
# 2. SUPERTREND — numpy (возвращает trend, up, dn)
# ======================================================================
def supertrend(df, period=10, mult=3.0):
    close = np.asarray(df['close'], dtype=float)
    high = np.asarray(df['high'], dtype=float)
    low = np.asarray(df['low'], dtype=float)
    n = len(close)
    atr_s = atr(high, low, close, period)
    src = (high + low) / 2.0
    up_basic = src - mult * atr_s
    dn_basic = src + mult * atr_s
    trend = np.zeros(n, dtype=int)
    up_prev = up_basic[0]; dn_prev = dn_basic[0]
    trend[0] = 1 if close[0] > up_basic[0] else -1
    for i in range(1, n):
        pc = close[i-1]
        if pc > up_prev:      up_new = max(up_basic[i], up_prev)
        else:                 up_new = up_basic[i]
        if pc < dn_prev:      dn_new = min(dn_basic[i], dn_prev)
        else:                 dn_new = dn_basic[i]
        if trend[i-1] == -1 and close[i] > dn_prev:   trend[i] = 1
        elif trend[i-1] == 1 and close[i] < up_prev:  trend[i] = -1
        else:                                             trend[i] = trend[i-1]
        up_prev, dn_prev = up_new, dn_new
    return trend, up_basic, dn_basic

print("\nСчитаю Supertrend (H1 + H4)...")
base_trend, st_up, st_dn = supertrend(base, ATR_PERIOD, ATR_MULTIPLIER)
htf_trend_raw, _, _ = supertrend(htf_df, ATR_PERIOD, ATR_MULTIPLIER)

# 🔴 ПРИВОДИМ HTF-ТРЕНД К ИНДЕКСУ BASE (ровно 50000, ffill)
htf_trend_h1 = pd.Series(htf_trend_raw, index=htf_df.index).reindex(idx_base).ffill().fillna(0).values

# ======================================================================
# 3. ЦЕЛЬ (производная Гаусса) + ФИЧИ
# ======================================================================
print("Генерирую фичи для LightGBM...")
close_arr = price_full
log_c = np.log(close_arr)
smoothed = gaussian_filter1d(log_c, SIGMA, mode='nearest')
deriv = np.gradient(smoothed)
roll_mean = pd.Series(deriv).rolling(500).mean().values
roll_std = pd.Series(deriv).rolling(500).std().values
norm = ((deriv - roll_mean) / (roll_std + 1e-8))

def make_features(df, prefix=''):
    c = np.asarray(df['close'], dtype=float)
    feat = pd.DataFrame(index=df.index)
    feat[f'{prefix}z20']  = (c - pd.Series(c).rolling(20).mean()) / (pd.Series(c).rolling(20).std() + 1e-8)
    feat[f'{prefix}z50']  = (c - pd.Series(c).rolling(50).mean()) / (pd.Series(c).rolling(50).std() + 1e-8)
    feat[f'{prefix}z100'] = (c - pd.Series(c).rolling(100).mean()) / (pd.Series(c).rolling(100).std() + 1e-8)
    feat[f'{prefix}z200'] = (c - pd.Series(c).rolling(200).mean()) / (pd.Series(c).rolling(200).std() + 1e-8)
    feat[f'{prefix}ret']  = pd.Series(c).pct_change()
    feat[f'{prefix}mom']  = pd.Series(c).diff().rolling(20).mean()
    feat[f'{prefix}vol30']  = pd.Series(c).pct_change().rolling(30).std()
    feat[f'{prefix}vol100'] = pd.Series(c).pct_change().rolling(100).std()
    idx = df.index
    feat[f'{prefix}sin_h']   = np.sin(2*np.pi*idx.hour/24)
    feat[f'{prefix}cos_h']   = np.cos(2*np.pi*idx.hour/24)
    feat[f'{prefix}sin_dow'] = np.sin(2*np.pi*idx.dayofweek/7)
    feat[f'{prefix}cos_dow'] = np.cos(2*np.pi*idx.dayofweek/7)
    return feat

# 🔴 ВСЕ фичи делаем сразу на idx_base (reindex + ffill)
feat = make_features(base, 'h1_')
for name, tf_df in [('h4_', htf_df), ('d_', daily_df)]:
    f = make_features(tf_df, name)
    f = f.reindex(idx_base).ffill()   # 🔴 ключевая строка — приводит к 50000
    feat = feat.join(f, how='left')

# ST-фичи (st_up/st_dn — уже 50000, base_trend — 50000)
st_up_s = pd.Series(st_up, index=idx_base).ffill().fillna(0).values
st_dn_s = pd.Series(st_dn, index=idx_base).ffill().fillna(0).values
feat['st_dist_up']  = (close_arr - st_up_s) / (close_arr + 1e-8)
feat['st_dist_dn']  = (close_arr - st_dn_s) / (close_arr + 1e-8)
feat['st_trend']    = base_trend
feat['htf_trend']   = htf_trend_h1

# каузальность T-1
feat = feat.shift(1).dropna()
y_series = pd.Series(norm, index=idx_base)
common = feat.index.intersection(y_series.dropna().index)
X = feat.loc[common]
y = y_series.loc[common]
print(f"Фичей: {X.shape[1]}, баров: {len(X)}")

# ======================================================================
# 4. БЭКТЕСТ (A/B/C) — чистые функции
# ======================================================================
def profit_lot(entry, exit_p, side, lot=LOT_SIZE):
    return (1 if side == 1 else -1) * (exit_p - entry) * 100000 * lot

def _close(pos_list, cur, typ, balance, log):
    p = sum(profit_lot(pos['entry'], cur, pos['side']) for pos in pos_list)
    balance += p
    log.append({'type': typ, 'profit': p, 'count': len(pos_list)})
    return [], balance

def _run(trend_arr, htf_arr, pred_lgbm, mode, price):
    n = len(trend_arr)
    buy_pos, sell_pos = [], []
    balance = INITIAL_BALANCE
    equity = [balance]
    log = []
    tp_amt = balance * (TP_PERCENT/100)
    risk_amt = -balance * (MAX_RISK_PERCENT/100)
    def bprof(pos, cur): return sum(profit_lot(p['entry'], cur, p['side']) for p in pos)

    for i in range(1, n):
        cur = price[i]
        tr = int(trend_arr[i])
        h = int(htf_arr[i])
        lgbm = pred_lgbm[i] if (pred_lgbm is not None and i < len(pred_lgbm)) else 0.0

        can_open = (not USE_HTF_FILTER) or (tr == h and h != 0)
        if mode in ('B', 'C'):
            can_open = can_open and ((tr == 1 and lgbm > LGBM_THRESHOLD) or
                                     (tr == -1 and lgbm < LGBM_THRESHOLD))
        if can_open and tr != 0 and (len(buy_pos)+len(sell_pos)) < MAX_POSITIONS_PER_SIDE*2:
            pos = {'entry': cur, 'side': tr}
            if tr == 1 and len(buy_pos) < MAX_POSITIONS_PER_SIDE:   buy_pos.append(pos)
            elif tr == -1 and len(sell_pos) < MAX_POSITIONS_PER_SIDE: sell_pos.append(pos)

        if mode == 'C' and pred_lgbm is not None:
            if len(buy_pos) > 0 and lgbm < LGBM_THRESHOLD:  buy_pos, balance = _close(buy_pos, cur, 'BUY_LGBM', balance, log)
            if len(sell_pos) > 0 and lgbm > LGBM_THRESHOLD: sell_pos, balance = _close(sell_pos, cur, 'SELL_LGBM', balance, log)

        if len(buy_pos) > 0 and bprof(buy_pos, cur) >= tp_amt:  buy_pos, balance = _close(buy_pos, cur, 'BUY_TP', balance, log)
        if len(sell_pos) > 0 and bprof(sell_pos, cur) >= tp_amt: sell_pos, balance = _close(sell_pos, cur, 'SELL_TP', balance, log)

        tot = bprof(buy_pos, cur) + bprof(sell_pos, cur)
        if tot <= risk_amt:
            balance += tot
            log.append({'type': 'GLOBAL_SL', 'profit': tot, 'count': len(buy_pos)+len(sell_pos)})
            buy_pos, sell_pos = [], []

        equity.append(balance + bprof(buy_pos, cur) + bprof(sell_pos, cur))

    bp = bprof(buy_pos, price[-1]); sp = bprof(sell_pos, price[-1]); balance += bp + sp
    if buy_pos or sell_pos:
        log.append({'type': 'FINAL', 'profit': bp+sp, 'count': len(buy_pos)+len(sell_pos)})

    log_df = pd.DataFrame(log)
    wr = (log_df['profit'] > 0).mean()*100 if len(log_df) else 0
    ret = (balance/INITIAL_BALANCE - 1)*100
    peak = np.maximum.accumulate(equity); mdd = ((peak - np.array(equity))/peak).max()*100
    return {'log': log_df, 'equity': np.array(equity), 'win_rate': wr, 'ret': ret, 'mdd': mdd}

# ======================================================================
# 5. WALK-FORWARD (A/B/C)
# ======================================================================
total = len(X); start = 0
results_A, results_B, results_C = [], [], []
all_idx = []

print(f"\n=== WALK-FORWARD HYBRID (A/B/C) ===")
while start + TRAIN_BARS + TEST_BARS <= total:
    te_start = start + TRAIN_BARS; te_end = te_start + TEST_BARS
    X_train, y_train = X.iloc[:te_start], y.iloc[:te_start]
    X_test, y_test = X.iloc[te_start:te_end], y.iloc[te_start:te_end]
    idx_test = list(X_test.index)

    price_test = price_full[te_start:te_end]
    trend_test = base_trend[te_start:te_end]
    htf_test = htf_trend_h1[te_start:te_end]

    model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31,
                              min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
                              reg_alpha=0.1, reg_lambda=0.1, objective='huber',
                              random_state=42, n_jobs=-1, verbose=-1)
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)],
              eval_metric='rmse', callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
    pred_test = model.predict(X_test).values
    corr = np.corrcoef(y_test.values, pred_test)[0, 1]

    rA = _run(trend_test, htf_test, None,     'A', price_test)
    rB = _run(trend_test, htf_test, pred_test, 'B', price_test)
    rC = _run(trend_test, htf_test, pred_test, 'C', price_test)

    results_A.append({'corr': corr, **rA}); results_B.append({'corr': corr, **rB}); results_C.append({'corr': corr, **rC})
    all_idx.extend(idx_test)
    print(f"  Окно {len(results_A)}: corr={corr:.3f} | A({rA['ret']:+.2f}%, MDD {rA['mdd']:.1f}) | "
          f"B({rB['ret']:+.2f}%, win {rB['win_rate']:.0f}%) | C({rC['ret']:+.2f}%, win {rC['win_rate']:.0f}%)")
    start += TEST_BARS

dfA = pd.DataFrame(results_A); dfB = pd.DataFrame(results_B); dfC = pd.DataFrame(results_C)

print("\n=== ИТОГО (среднее по окнам) ===")
summary = pd.DataFrame({
    'Variant': ['A: Supertrend (безопасный)', 'B: + LGBM фильтр входов', 'C: Гибрид (LGBM-выход)'],
    'Ret_%': [dfA['ret'].mean(), dfB['ret'].mean(), dfC['ret'].mean()],
    'Win_%': [dfA['win_rate'].mean(), dfB['win_rate'].mean(), dfC['win_rate'].mean()],
    'MDD_%': [dfA['mdd'].mean(), dfB['mdd'].mean(), dfC['mdd'].mean()],
    'Corr':  [dfA['corr'].mean(), dfB['corr'].mean(), dfC['corr'].mean()],
})
print(summary.round(2).to_string(index=False))

# ======================================================================
# 6. ГРАФИКИ
# ======================================================================
L = len(all_idx)
def al(x): return np.array(x)[:L]

fig, axes = plt.subplots(3, 1, figsize=(16, 15))
ax = axes[0]
y_s = pd.Series(norm, index=idx_base).loc[all_idx]
ax.plot(al(all_idx), al(y_s.values), color='steelblue', lw=0.7, label='Цель (производная Гаусса)')
ax.axhline(0, color='gray', lw=0.8); ax.set_title('1) Цель (Ground Truth)'); ax.legend(); ax.grid(alpha=0.3)

ax = axes[1]
bt = pd.Series(base_trend, index=idx_base).loc[all_idx]
ax.plot(al(all_idx), al(bt.values), color='purple', lw=0.6, label='Supertrend trend')
ax.fill_between(al(all_idx), 0, al(bt.values), where=(al(bt.values)>0), color='green', alpha=0.15)
ax.fill_between(al(all_idx), 0, al(bt.values), where=(al(bt.values)<0), color='red', alpha=0.15)
ax.axhline(0, color='gray', lw=0.8); ax.set_title('2) Supertrend + LGBM (гибрид C)'); ax.legend(); ax.grid(alpha=0.3)

ax = axes[2]
for df, name, c in [(dfA, 'A (оригинал)', 'gray'), (dfB, 'B (фильтр)', 'orange'), (dfC, 'C (гибрид)', 'green')]:
    eq = np.concatenate([r['equity'] for r in df['equity']])
    eq = 10000 * eq / eq[0]
    ax.plot(al(all_idx), al(eq), color=c, lw=1, label=f"{name}: {df['ret'].mean():+.1f}% (MDD {df['mdd'].mean():.1f}%)")
ax.axhline(10000, color='gray', ls=':', alpha=0.7); ax.set_title('3) Equity: A vs B vs C (нормировано)'); ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout(); plt.savefig('chart_hybrid.png', dpi=120)
print("\n✅ Сохранён: chart_hybrid.png"); plt.show()
print("\n✅ Отключаюсь от MT5."); mt5.shutdown()