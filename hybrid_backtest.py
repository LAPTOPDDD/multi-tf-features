import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import glob, os

np.random.seed(42)

# ======================================================================
# ПАРАМЕТРЫ (Вариант: одно окно 200 баров)
# ======================================================================
SYMBOL = "EURUSD"

# Таймфреймы (базовый = H1, старшие = H4 + DAILY)
BASE_TF = "H1"
HTF = "H4"
DAILY_TF = "DAILY"

ATR_PERIOD = 10
ATR_MULTIPLIER = 3.0
USE_HTF_FILTER = True

LOT_SIZE = 0.05
INITIAL_BALANCE = 10000.0
TP_PERCENT = 0.5
MAX_RISK_PERCENT = 15.0
MAX_POSITIONS_PER_SIDE = 25

SIGMA = 5
LGBM_THRESHOLD = 0.0

# ОДНО ОКНО: 4000 баров обучения, 200 баров теста
TRAIN_BARS = 4000
TEST_BARS =2000
DATA_DIR = ""

# ======================================================================
# 0. ЗАГРУЗКА ДАННЫХ
# ======================================================================
def find_file(pattern):
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = glob.glob(os.path.join(here, pattern)) + glob.glob(os.path.join(DATA_DIR, pattern))
    if not candidates:
        raise FileNotFoundError(f"Не найден файл по маске: {pattern}\nПоложи CSV рядом со скриптом.")
    return candidates[0]

def load_csv(path, tf_name):
    # Пробуем разные разделители
    for sep in ['\t', ';', ',']:
        try:
            df = pd.read_csv(path, sep=sep, dtype=str, encoding='utf-8')
            break
        except:
            continue
    else:
        raise ValueError(f"{tf_name}: не удалось прочитать файл")
    
    # Приводим названия колонок к верхнему регистру и убираем лишние символы
    df.columns = [c.replace('<', '').replace('>', '').replace(' ', '').upper() for c in df.columns]
    
    # Ищем TIME и DATE
    time_col = None
    date_col = None
    
    for col in df.columns:
        if 'TIME' in col:
            time_col = col
        if 'DATE' in col:
            date_col = col
    
    # Создаем datetime индекс
    if time_col and date_col:
        df['datetime'] = pd.to_datetime(df[date_col] + ' ' + df[time_col], errors='coerce')
    elif date_col:
        df['datetime'] = pd.to_datetime(df[date_col], errors='coerce')
    else:
        raise ValueError(f"{tf_name}: нет колонок TIME и/или DATE")
    
    df = df.dropna(subset=['datetime']).set_index('datetime').sort_index()
    
    # Ищем OHLC колонки
    need = {'OPEN': 'open', 'HIGH': 'high', 'LOW': 'low', 'CLOSE': 'close'}
    ren = {}
    for k, v in need.items():
        found = False
        for col in df.columns:
            if k in col:
                ren[col] = v
                found = True
                break
        if not found:
            raise ValueError(f"{tf_name}: нет колонки {k}")
    
    df = df.rename(columns=ren)[['open', 'high', 'low', 'close']]
    
    # Конвертируем в числа
    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna()
    
    # Оставляем только будни и сортируем
    df = df[df.index.dayofweek < 5].sort_index()
    print(f"  {tf_name}: {len(df)} баров  [{df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}]")
    return df

print("Ищу CSV-файлы рядом со скриптом...")
base_path = find_file(f"*{SYMBOL}_{BASE_TF}*.csv")
htf_path  = find_file(f"*{SYMBOL}_{HTF}*.csv")
d_path    = find_file(f"*{SYMBOL}_{DAILY_TF}*.csv")

print(f"Найденные файлы:")
print(f"  BASE: {os.path.basename(base_path)}")
print(f"  HTF:  {os.path.basename(htf_path)}")
print(f"  DAILY: {os.path.basename(d_path)}")

print("\nЗагружаю данные (базовый H1 + H4 + DAILY)...")
base    = load_csv(base_path, BASE_TF)
htf_df  = load_csv(htf_path, HTF)
daily_df = load_csv(d_path, DAILY_TF)

price_full = base['close'].values.copy()
idx_base = base.index

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
# 2. SUPERTREND — numpy
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
    up_prev, dn_prev = up_basic[0], dn_basic[0]
    trend[0] = 1 if close[0] > up_basic[0] else -1
    for i in range(1, n):
        pc = close[i-1]
        up_new = max(up_basic[i], up_prev) if pc > up_prev else up_basic[i]
        dn_new = min(dn_basic[i], dn_prev) if pc < dn_prev else dn_basic[i]
        if trend[i-1] == -1 and close[i] > dn_prev:   trend[i] = 1
        elif trend[i-1] == 1 and close[i] < up_prev:  trend[i] = -1
        else:                                             trend[i] = trend[i-1]
        up_prev, dn_prev = up_new, dn_new
    return trend, up_basic, dn_basic

print("\nСчитаю Supertrend (H1 + H4)...")
base_trend, st_up, st_dn = supertrend(base, ATR_PERIOD, ATR_MULTIPLIER)
htf_trend_raw, _, _ = supertrend(htf_df, ATR_PERIOD, ATR_MULTIPLIER)
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
    cs = pd.Series(c, index=df.index)
    feat = pd.DataFrame(index=df.index)
    for p in [20, 50, 100, 200]:
        m = cs.rolling(p).mean(); s = cs.rolling(p).std()
        feat[f'{prefix}z{p}'] = (c - m.values) / (s.values + 1e-8)
    feat[f'{prefix}ret'] = cs.pct_change().values
    feat[f'{prefix}mom'] = cs.diff().rolling(20).mean().values
    feat[f'{prefix}vol30'] = cs.pct_change().rolling(30).std().values
    feat[f'{prefix}vol100'] = cs.pct_change().rolling(100).std().values
    idx = df.index
    feat[f'{prefix}sin_h'] = np.sin(2*np.pi*idx.hour/24)
    feat[f'{prefix}cos_h'] = np.cos(2*np.pi*idx.hour/24)
    feat[f'{prefix}sin_dow'] = np.sin(2*np.pi*idx.dayofweek/7)
    feat[f'{prefix}cos_dow'] = np.cos(2*np.pi*idx.dayofweek/7)
    return feat

print("  ресемплю H4 + DAILY на H1...")
feat = make_features(base, 'h1_')
for name, tf_df in [('h4_', htf_df), ('d_', daily_df)]:
    f = make_features(tf_df, name)
    f = f.reindex(idx_base).ffill()
    feat = feat.join(f, how='left')

st_up_s = pd.Series(st_up, index=idx_base).ffill().fillna(0).values
st_dn_s = pd.Series(st_dn, index=idx_base).ffill().fillna(0).values
feat['st_dist_up'] = (close_arr - st_up_s) / (close_arr + 1e-8)
feat['st_dist_dn'] = (close_arr - st_dn_s) / (close_arr + 1e-8)
feat['st_trend'] = base_trend
feat['htf_trend'] = htf_trend_h1

feat = feat.shift(1).dropna()
y_series = pd.Series(norm, index=idx_base)
common = feat.index.intersection(y_series.dropna().index)
X = feat.loc[common]
y = y_series.loc[common]
print(f"Фичей: {X.shape[1]}, баров: {len(X)}")

# ======================================================================
# 4. БЭКТЕСТ (A/B/C)
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
# 5. ОДНО ОКНО WALK-FORWARD (200 баров теста)
# ======================================================================
total = len(X)

# Берем только одно окно: обучаем на первых TRAIN_BARS, тестируем на следующих TEST_BARS
te_start = TRAIN_BARS
te_end = te_start + TEST_BARS

if te_end > total:
    print(f"⚠️ Недостаточно данных: нужно {te_end}, есть {total}")
    print(f"Использую последние {TEST_BARS} баров для теста")
    te_start = total - TEST_BARS
    te_end = total

Xtr, ytr = X.iloc[:te_start], y.iloc[:te_start]
Xte, yte = X.iloc[te_start:te_end], y.iloc[te_start:te_end]
idx_test = list(Xte.index)

price_test = price_full[te_start:te_end]
trend_test = base_trend[te_start:te_end]
htf_test = htf_trend_h1[te_start:te_end]

print(f"\n=== ОДНО ОКНО: обучение {len(Xtr)} баров, тест {len(Xte)} баров ===")

model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31,
                          min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
                          reg_alpha=0.1, reg_lambda=0.1, objective='huber',
                          random_state=42, n_jobs=-1, verbose=-1)
model.fit(Xtr, ytr,
          eval_set=[(Xtr, ytr), (Xte, yte)],
          eval_metric='rmse',
          callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])

pred_test = model.predict(Xte)
corr = np.corrcoef(yte.values, pred_test)[0, 1]

rA = _run(trend_test, htf_test, None,     'A', price_test)
rB = _run(trend_test, htf_test, pred_test, 'B', price_test)
rC = _run(trend_test, htf_test, pred_test, 'C', price_test)

print(f"\nРезультаты:")
print(f"  Corr: {corr:.3f}")
print(f"  A (оригинал):  {rA['ret']:+.2f}%, MDD {rA['mdd']:.2f}%, WinRate {rA['win_rate']:.1f}%")
print(f"  B (фильтр):    {rB['ret']:+.2f}%, MDD {rB['mdd']:.2f}%, WinRate {rB['win_rate']:.1f}%")
print(f"  C (гибрид):    {rC['ret']:+.2f}%, MDD {rC['mdd']:.2f}%, WinRate {rC['win_rate']:.1f}%")

# ======================================================================
# 6. ГРАФИКИ
# ======================================================================
fig, axes = plt.subplots(3, 1, figsize=(16, 15))

# 1) Цель
ax = axes[0]
y_s = pd.Series(norm, index=idx_base).loc[idx_test]
ax.plot(idx_test, y_s.values, color='steelblue', lw=0.8, label='Цель LGBM (производная Гаусса)')
ax.axhline(0, color='gray', lw=0.8)
ax.set_title('1) Цель (Ground Truth) — H1')
ax.legend()
ax.grid(alpha=0.3)

# 2) Supertrend
ax = axes[1]
bt = pd.Series(base_trend, index=idx_base).loc[idx_test]
ax.plot(idx_test, bt.values, color='purple', lw=0.6, label='Supertrend тренд (H1)')
ax.fill_between(idx_test, 0, bt.values, where=(bt.values>0), color='green', alpha=0.15)
ax.fill_between(idx_test, 0, bt.values, where=(bt.values<0), color='red', alpha=0.15)
ax.axhline(0, color='gray', lw=0.8)
ax.set_title('2) Supertrend тренд')
ax.legend()
ax.grid(alpha=0.3)

# 3) Equity
ax = axes[2]
eq_A = rA['equity']
eq_B = rB['equity']
eq_C = rC['equity']

eq_A = 10000 * eq_A / eq_A[0]
eq_B = 10000 * eq_B / eq_B[0]
eq_C = 10000 * eq_C / eq_C[0]

ax.plot(idx_test, eq_A, color='gray', lw=1.5, label=f"A (оригинал): {rA['ret']:+.1f}%")
ax.plot(idx_test, eq_B, color='orange', lw=1.5, label=f"B (фильтр): {rB['ret']:+.1f}%")
ax.plot(idx_test, eq_C, color='green', lw=1.5, label=f"C (гибрид): {rC['ret']:+.1f}%")
ax.axhline(10000, color='gray', ls=':', alpha=0.7)
ax.set_title('3) Equity: A vs B vs C (нормировано к 10000)')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('chart_hybrid_H1_single.png', dpi=120)
print("\n✅ Сохранён: chart_hybrid_H1_single.png")
plt.show()