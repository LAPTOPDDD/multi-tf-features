import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy import stats
import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import glob, os

np.random.seed(42)

# ======================================================================
# ПАРАМЕТРЫ
# ======================================================================
SYMBOL = "EURUSD"
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

TRAIN_BARS = 4000
TEST_BARS = 200
DATA_DIR = ""

# ======================================================================
# 0. ЗАГРУЗКА ДАННЫХ
# ======================================================================
def find_file(pattern):
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = glob.glob(os.path.join(here, pattern)) + glob.glob(os.path.join(DATA_DIR, pattern))
    if not candidates:
        raise FileNotFoundError(f"Не найден файл по маске: {pattern}")
    return candidates[0]

def load_csv(path, tf_name):
    for sep in ['\t', ';', ',']:
        try:
            df = pd.read_csv(path, sep=sep, dtype=str, encoding='utf-8')
            break
        except:
            continue
    else:
        raise ValueError(f"{tf_name}: не удалось прочитать файл")
    
    df.columns = [c.replace('<', '').replace('>', '').replace(' ', '').upper() for c in df.columns]
    
    time_col = None
    date_col = None
    for col in df.columns:
        if 'TIME' in col:
            time_col = col
        if 'DATE' in col:
            date_col = col
    
    if time_col and date_col:
        df['datetime'] = pd.to_datetime(df[date_col] + ' ' + df[time_col], errors='coerce')
    elif date_col:
        df['datetime'] = pd.to_datetime(df[date_col], errors='coerce')
    else:
        raise ValueError(f"{tf_name}: нет колонок TIME и/или DATE")
    
    df = df.dropna(subset=['datetime']).set_index('datetime').sort_index()
    
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
    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    df = df[df.index.dayofweek < 5].sort_index()
    print(f"  {tf_name}: {len(df)} баров")
    return df

print("Загружаю данные...")
base_path = find_file(f"*{SYMBOL}_{BASE_TF}*.csv")
htf_path  = find_file(f"*{SYMBOL}_{HTF}*.csv")
d_path    = find_file(f"*{SYMBOL}_{DAILY_TF}*.csv")

base    = load_csv(base_path, BASE_TF)
htf_df  = load_csv(htf_path, HTF)
daily_df = load_csv(d_path, DAILY_TF)

price_full = base['close'].values.copy()
idx_base = base.index

# ======================================================================
# 1. БАЗОВЫЕ ФУНКЦИИ (ATR, Supertrend)
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

print("\nСчитаю Supertrend...")
base_trend, st_up, st_dn = supertrend(base, ATR_PERIOD, ATR_MULTIPLIER)
htf_trend_raw, _, _ = supertrend(htf_df, ATR_PERIOD, ATR_MULTIPLIER)
htf_trend_h1 = pd.Series(htf_trend_raw, index=htf_df.index).reindex(idx_base).ffill().fillna(0).values

# ======================================================================
# 2. ЦЕЛЬ И ФИЧИ
# ======================================================================
print("Генерирую фичи...")
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
# 3. СТАТИСТИЧЕСКИЙ АНАЛИЗ АВТОКОРРЕЛЯЦИИ
# ======================================================================
print("\n" + "="*60)
print("СТАТИСТИЧЕСКИЙ АНАЛИЗ АВТОКОРРЕЛЯЦИИ")
print("="*60)

def correlation_analysis(data, series_name="Временной ряд"):
    """
    Комплексный анализ автокорреляции
    """
    n = len(data)
    results = {}
    
    # 1. Автокорреляционная функция (ACF) для разных лагов
    print(f"\n--- АВТОКОРРЕЛЯЦИЯ: {series_name} ---")
    max_lag = min(30, n // 3)
    autocorr = []
    significant_lags = []
    
    for lag in range(1, max_lag + 1):
        if lag >= n:
            break
        corr, p_value = stats.pearsonr(data[:-lag], data[lag:])
        autocorr.append((lag, corr, p_value))
        
        # Проверяем значимость (p < 0.05)
        if abs(corr) > 0.05 and p_value < 0.05:
            significant_lags.append(lag)
            print(f"  Лаг {lag:2d}: r = {corr:+.4f} (p={p_value:.4f}) {'*' if p_value < 0.05 else ''}")
        elif abs(corr) > 0.05:
            print(f"  Лаг {lag:2d}: r = {corr:+.4f} (p={p_value:.4f})")
    
    # 2. Суммарная статистика
    print(f"\n  ✓ Значимых лагов: {len(significant_lags)} из {max_lag}")
    if significant_lags:
        print(f"  ✓ Первые значимые лаги: {significant_lags[:10]}")
    
    # 3. Ljung-Box тест (проверка на наличие автокорреляции в целом)
    from statsmodels.stats.diagnostic import acf, q_stat
    import statsmodels.api as sm
    
    # Используем функцию acf из statsmodels
    acf_vals = sm.tsa.acf(data, nlags=min(20, n//2), fft=False)
    
    # Q-статистика Льюнга-Бокса
    q_statistic, p_value_q = sm.tsa.acf(data, nlags=min(20, n//2), fft=False)
    
    # 4. Durbin-Watson статистика (проверка на автокорреляцию 1-го порядка)
    from statsmodels.stats.stattools import durbin_watson
    dw = durbin_watson(data)
    print(f"\n  ✓ Durbin-Watson: {dw:.3f} (значения <1.0 или >3.0 указывают на автокорреляцию)")
    
    # 5. Частичная автокорреляция (PACF)
    from statsmodels.tsa.stattools import pacf
    pacf_vals = pacf(data, nlags=min(20, n//2))
    
    # 6. Сохраняем результаты
    results['autocorr'] = autocorr
    results['significant_lags'] = significant_lags
    results['pacf'] = pacf_vals
    results['acf'] = acf_vals
    results['durbin_watson'] = dw
    
    return results

# Анализируем разные ряды
print("\n1. ЦЕНА (закрытие):")
price_data = close_arr[-5000:]  # берем последние 5000 баров
price_corr = correlation_analysis(price_data, "Цена")

print("\n2. ДОХОДНОСТЬ (returns):")
returns = np.diff(np.log(price_data))
returns_corr = correlation_analysis(returns, "Доходность")

print("\n3. ЦЕЛЬ (производная Гаусса):")
target_data = norm[-5000:]
target_corr = correlation_analysis(target_data, "Цель")

print("\n4. ТРЕНД SUPERTREND:")
trend_data = base_trend[-5000:]
trend_corr = correlation_analysis(trend_data, "Supertrend")

# ======================================================================
# 4. ВИЗУАЛИЗАЦИЯ АВТОКОРРЕЛЯЦИИ
# ======================================================================
fig, axes = plt.subplots(4, 2, figsize=(16, 14))

# 1. Графики рядов
series_list = [
    (price_data[-500:], "Цена (последние 500)", axes[0, 0]),
    (returns[-500:], "Доходность (последние 500)", axes[1, 0]),
    (target_data[-500:], "Цель (последние 500)", axes[2, 0]),
    (trend_data[-500:], "Supertrend (последние 500)", axes[3, 0])
]

for i, (data, title, ax) in enumerate(series_list):
    ax.plot(data, color='steelblue', lw=0.8)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.axhline(0, color='gray', lw=0.5, alpha=0.5)

# 2. Автокорреляционные функции
acf_list = [
    (price_corr['acf'], "ACF: Цена", axes[0, 1]),
    (returns_corr['acf'], "ACF: Доходность", axes[1, 1]),
    (target_corr['acf'], "ACF: Цель", axes[2, 1]),
    (trend_corr['acf'], "ACF: Supertrend", axes[3, 1])
]

for i, (acf_vals, title, ax) in enumerate(acf_list):
    ax.bar(range(len(acf_vals)), acf_vals, color='steelblue', alpha=0.7)
    ax.axhline(0, color='black', lw=0.5)
    ax.axhline(1.96/np.sqrt(len(acf_vals)), color='red', ls='--', lw=0.8, label='95% ДИ')
    ax.axhline(-1.96/np.sqrt(len(acf_vals)), color='red', ls='--', lw=0.8)
    ax.set_title(title)
    ax.set_xlabel("Лаг")
    ax.grid(alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.savefig('autocorrelation_analysis.png', dpi=120)
print("\n✅ Сохранён: autocorrelation_analysis.png")

# ======================================================================
# 5. КОРРЕЛЯЦИЯ МЕЖДУ ТЕКУЩИМ СОСТОЯНИЕМ И ПРЕДЫДУЩИМИ
# ======================================================================
print("\n" + "="*60)
print("КОРРЕЛЯЦИЯ МЕЖДУ ТЕКУЩИМ И ПРЕДЫДУЩИМИ СОСТОЯНИЯМИ")
print("="*60)

def lag_correlation_analysis(data, max_lag=30):
    """Анализ корреляции между текущим значением и значениями с разными лагами"""
    results = []
    n = len(data)
    
    for lag in range(1, min(max_lag + 1, n - 1)):
        # Текущее состояние (y) и предыдущее (x) с лагом
        y = data[lag:]
        x = data[:-lag]
        
        # Pearson корреляция
        pearson_r, p_value = stats.pearsonr(x, y)
        
        # Spearman корреляция (ранговая)
        spearman_r, p_spearman = stats.spearmanr(x, y)
        
        # Kendall Tau
        kendall_tau, p_kendall = stats.kendalltau(x, y)
        
        results.append({
            'lag': lag,
            'pearson': pearson_r,
            'pearson_p': p_value,
            'spearman': spearman_r,
            'kendall': kendall_tau
        })
    
    return pd.DataFrame(results)

# Анализируем для разных рядов
for name, data in [
    ("Цена", price_data),
    ("Доходность", returns),
    ("Цель", target_data)
]:
    print(f"\n--- {name} ---")
    df_corr = lag_correlation_analysis(data, max_lag=20)
    
    # Показываем топ-5 значимых корреляций
    significant = df_corr[abs(df_corr['pearson']) > 0.05]
    if len(significant) > 0:
        top = significant.nlargest(5, 'pearson')
        for _, row in top.iterrows():
            print(f"  Лаг {int(row['lag']):2d}: P={row['pearson']:+.4f}, S={row['spearman']:+.4f}")
    
    # Суммарная статистика
    strong_corr = df_corr[abs(df_corr['pearson']) > 0.1]
    print(f"  Значимых корреляций (>0.1): {len(strong_corr)} из {len(df_corr)}")

# ======================================================================
# 6. БЭКТЕСТ
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
# 7. ОДНО ОКНО
# ======================================================================
total = len(X)
te_start = TRAIN_BARS
te_end = te_start + TEST_BARS

if te_end > total:
    print(f"\n⚠️ Недостаточно данных: нужно {te_end}, есть {total}")
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

print(f"\nРезультаты бэктеста:")
print(f"  Corr: {corr:.3f}")
print(f"  A (оригинал):  {rA['ret']:+.2f}%, MDD {rA['mdd']:.2f}%, WinRate {rA['win_rate']:.1f}%")
print(f"  B (фильтр):    {rB['ret']:+.2f}%, MDD {rB['mdd']:.2f}%, WinRate {rB['win_rate']:.1f}%")
print(f"  C (гибрид):    {rC['ret']:+.2f}%, MDD {rC['mdd']:.2f}%, WinRate {rC['win_rate']:.1f}%")

# ======================================================================
# 8. ИТОГОВЫЙ ГРАФИК
# ======================================================================
fig2, axes2 = plt.subplots(3, 1, figsize=(16, 15))

ax = axes2[0]
y_s = pd.Series(norm, index=idx_base).loc[idx_test]
ax.plot(idx_test, y_s.values, color='steelblue', lw=0.8, label='Цель LGBM')
ax.axhline(0, color='gray', lw=0.8)
ax.set_title('Цель (Ground Truth)')
ax.legend()
ax.grid(alpha=0.3)

ax = axes2[1]
bt = pd.Series(base_trend, index=idx_base).loc[idx_test]
ax.plot(idx_test, bt.values, color='purple', lw=0.6, label='Supertrend')
ax.fill_between(idx_test, 0, bt.values, where=(bt.values>0), color='green', alpha=0.15)
ax.fill_between(idx_test, 0, bt.values, where=(bt.values<0), color='red', alpha=0.15)
ax.axhline(0, color='gray', lw=0.8)
ax.set_title('Supertrend тренд')
ax.legend()
ax.grid(alpha=0.3)

ax = axes2[2]
eq_A = 10000 * rA['equity'] / rA['equity'][0]
eq_B = 10000 * rB['equity'] / rB['equity'][0]
eq_C = 10000 * rC['equity'] / rC['equity'][0]

ax.plot(idx_test, eq_A, color='gray', lw=1.5, label=f"A (оригинал): {rA['ret']:+.1f}%")
ax.plot(idx_test, eq_B, color='orange', lw=1.5, label=f"B (фильтр): {rB['ret']:+.1f}%")
ax.plot(idx_test, eq_C, color='green', lw=1.5, label=f"C (гибрид): {rC['ret']:+.1f}%")
ax.axhline(10000, color='gray', ls=':', alpha=0.7)
ax.set_title('Equity: A vs B vs C')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('backtest_results.png', dpi=120)
print("\n✅ Сохранён: backtest_results.png")
print("\n🎯 Анализ завершён!")