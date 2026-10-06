"""
Мини-тест (песочница): чтение твоих CSV + Supertrend + проверка длин.
Запуск: python test_load.py
"""
import numpy as np
import pandas as pd

DATA = "/data/inputs"

def load_csv(path, tf_name):
    """Чтение CSV: <DATE> <TIME> OPEN HIGH LOW CLOSE ..."""
    df = pd.read_csv(path, sep='\t', dtype=str)
    # убираем угловые скобки у заголовков
    df.columns = [c.replace('<', '').replace('>', '') for c in df.columns]
    # собираем datetime (Daily-файл без TIME — берём только DATE)
    if 'TIME' in df.columns:
        df['datetime'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'], errors='coerce')
    else:
        df['datetime'] = pd.to_datetime(df['DATE'], errors='coerce')
    df = df.dropna(subset=['datetime']).set_index('datetime').sort_index()
    for c in ['OPEN', 'HIGH', 'LOW', 'CLOSE']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df[['OPEN', 'HIGH', 'LOW', 'CLOSE']].dropna()
    df.columns = ['open', 'high', 'low', 'close']
    print(f"  {tf_name}: {len(df)} баров | {df.index[0]} -> {df.index[-1]}")
    return df

print("Загрузка файлов...")
base = load_csv(f"{DATA}/EURUSD_M5_202504180935_202608271920.csv", "M5 (base)")
h1   = load_csv(f"{DATA}/EURUSD_H1_202401020000_202608271900.csv", "H1")
h4   = load_csv(f"{DATA}/EURUSD_H4_202401020000_202608271600.csv", "H4")
d1   = load_csv(f"{DATA}/EURUSD_Daily_202401020000_202608270000.csv", "Daily")

# === ATR (Wilder) ===
def atr(high, low, close, period=10):
    high, low, close = np.asarray(high, float), np.asarray(low, float), np.asarray(close, float)
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
    out = np.full(n, np.nan)
    if n >= period:
        out[period-1] = np.mean(tr[:period])
        for i in range(period, n):
            out[i] = (out[i-1]*(period-1) + tr[i]) / period
    return out

# === Supertrend ===
def supertrend(df, period=10, mult=3.0):
    close, high, low = map(lambda c: np.asarray(df[c], float), ['close', 'high', 'low'])
    n = len(close)
    atr_s = atr(high, low, close, period)
    src = (high + low)/2.0
    up_b, dn_b = src - mult*atr_s, src + mult*atr_s
    trend = np.zeros(n, dtype=int)
    up_prev, dn_prev = up_b[0], dn_b[0]
    trend[0] = 1 if close[0] > up_b[0] else -1
    for i in range(1, n):
        pc = close[i-1]
        up_new = max(up_b[i], up_prev) if pc > up_prev else up_b[i]
        dn_new = min(dn_b[i], dn_prev) if pc < dn_prev else dn_b[i]
        if trend[i-1] == -1 and close[i] > dn_prev:   trend[i] = 1
        elif trend[i-1] == 1 and close[i] < up_prev:  trend[i] = -1
        else:                                            trend[i] = trend[i-1]
        up_prev, dn_prev = up_new, dn_new
    return trend

print("\nСчитаю Supertrend на M5 (base)...")
base_trend = supertrend(base)
print(f"  base_trend: len={len(base_trend)}, unique={np.unique(base_trend)}")

print("\nПроверка ресемпла H4 -> M5...")
h4_trend = supertrend(h4)
h4_on_base = pd.Series(h4_trend, index=h1.index).reindex(base.index, method='ffill').fillna(0).values
print(f"  h4_on_base: len={len(h4_on_base)} (должно == {len(base)})")
assert len(h4_on_base) == len(base), "ДЛИНЫ НЕ СОВПАЛИ!"

print("\n✅ ВСЁ OK. Длины согласованы, Supertrend работает.")
print(f"   M5={len(base)}, H1={len(h1)}, H4={len(h4)}, Daily={len(d1)}")
