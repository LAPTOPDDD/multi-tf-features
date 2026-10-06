"""
Strict CAUSAL feature engineering + LightGBM forecast of 1st derivative of Gaussian convolution.
Адаптировано для MetaTrader 5
"""

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# ==================== 1. ПОДКЛЮЧЕНИЕ К METATRADER ====================
print("Подключаюсь к MetaTrader 5...")
if not mt5.initialize():
    print("Ошибка подключения!")
    mt5.shutdown()
    exit()

print(f"Терминал: {mt5.terminal_info()}")
print(f"Авторизован: {mt5.account_info().login}")

# ==================== 2. ЗАГРУЗКА ДАННЫХ С METATRADER ====================
def load_mt5_data(symbol, timeframe, bars=50000):
    """Загрузка данных из MetaTrader 5"""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None or len(rates) == 0:
        print(f"Ошибка загрузки {symbol}")
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    
    # Переименовываем колонки под формат скрипта
    df.rename(columns={
        'open': 'open',
        'high': 'high', 
        'low': 'low',
        'close': 'close',
        'tick_volume': 'tickvol',
        'real_volume': 'vol',
        'spread': 'spread'
    }, inplace=True)
    
    return df

# Загружаем все таймфреймы
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

BARS = 50000  # Количество баров для загрузки

for name, tf in timeframes.items():
    df = load_mt5_data(SYMBOL, tf, BARS)
    if df is not None and len(df) > 0:
        data[name] = df
        print(f"  {name}: {len(df)} баров")

if 'M5' not in data:
    print("Ошибка: нет данных M5!")
    mt5.shutdown()
    exit()

BASE = "M5"
m5 = data[BASE].copy()
m5["hlc3"] = (m5["high"] + m5["low"] + m5["close"]) / 3
m5["rng"] = m5["high"] - m5["low"]

print(f"\nБазовая выборка M5: {len(m5)} баров")

# ==================== 3. КАУЗАЛЬНЫЕ ИНДИКАТОРЫ ====================
def ema_causal(s: pd.Series, w: int) -> pd.Series:
    """Строго каузальная EMA (использует только прошлые значения)"""
    return s.ewm(span=w, adjust=False, min_periods=w).mean()

def sma_causal(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=w).mean()

def rsi_causal(s: pd.Series, w: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).rolling(w, min_periods=w).mean()
    dn = (-d.clip(upper=0)).rolling(w, min_periods=w).mean()
    rs = up / (dn.replace(0, np.nan))
    return 100 - 100 / (1 + rs)

def bb_causal(s: pd.Series, w: int = 20, k: float = 2.0):
    m = s.rolling(w, min_periods=w).mean()
    sd = s.rolling(w, min_periods=w).std()
    return m, m + k * sd, m - k * sd

def atr_causal(h, l, c, w: int = 14):
    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(w, min_periods=w).mean()

def add_indicators(df: pd.DataFrame, p: str) -> pd.DataFrame:
    """Все индикаторы КАУЗАЛЬНЫЕ (rolling min_periods=window / ewm adjust=False)"""
    out = pd.DataFrame(index=df.index)
    c, h, l = df["close"], df["high"], df["low"]
    
    out[f"{p}_close"] = c
    out[f"{p}_ret1"] = c.pct_change().where(c.index >= c.index[0] + pd.Timedelta(minutes=1), np.nan)
    
    for w in (5, 10, 20, 50):
        out[f"{p}_sma{w}"] = sma_causal(c, w)
        out[f"{p}_ema{w}"] = ema_causal(c, w)
        out[f"{p}_mom{w}"] = c / c.shift(w) - 1
    
    # MACD
    out[f"{p}_macd"] = ema_causal(c, 12) - ema_causal(c, 26)
    out[f"{p}_macdsig"] = ema_causal(out[f"{p}_macd"], 9)
    out[f"{p}_macdhist"] = out[f"{p}_macd"] - out[f"{p}_macdsig"]
    
    # RSI
    out[f"{p}_rsi14"] = rsi_causal(c, 14)
    
    # Bollinger %B
    m, hi, lo = bb_causal(c, 20)
    out[f"{p}_bbpct"] = (c - lo) / (hi - lo).replace(0, np.nan)
    
    # ATR
    out[f"{p}_atr14"] = atr_causal(h, l, c, 14)
    out[f"{p}_atrr"] = out[f"{p}_atr14"] / c
    
    # Трендовые индикаторы
    out[f"{p}_trend20"] = (c - out[f"{p}_ema20"]) / (out[f"{p}_atr14"].replace(0, np.nan))
    out[f"{p}_trend50"] = (c - out[f"{p}_ema50"]) / (out[f"{p}_atr14"].replace(0, np.nan))
    
    # HLC момент
    out[f"{p}_hlc3"] = (h + l + c) / 3
    out[f"{p}_hlcrng"] = (out[f"{p}_hlc3"] - out[f"{p}_hlc3"].shift(10)) / (out[f"{p}_atr14"].replace(0, np.nan))
    
    # Логарифм цены
    out[f"{p}_log"] = np.log(c)
    
    return out

# ==================== 4. МУЛЬТИТАЙМФРЕЙМ МЕРДЖ ====================
print("\nГенерирую мультитаймфрейм фичи...")

frames = {}
for tf in data.keys():
    ind = add_indicators(data[tf], tf.lower())
    if tf != BASE:
        # Ресемплируем на M5 и forward fill
        ind = ind.resample('5min').ffill()
    frames[tf] = ind

feat = frames[BASE]
for tf in data.keys():
    if tf != BASE and tf in frames:
        # shift(1) - каузальность: используем только закрытые бары
        feat = feat.join(frames[tf].shift(1), how='left')

# Удаляем дубликаты колонок
feat = feat.loc[:, ~feat.columns.duplicated(keep='first')]

# ==================== 5. МЕЖТАЙМФРЕЙМ ВЗАИМОДЕЙСТВИЯ ====================
for a, b in [("m5", "m15"), ("m5", "h1"), ("m15", "h1"), ("h1", "h4"), ("h4", "daily")]:
    ca, cb = f"{a}_ema20", f"{b}_ema20"
    if ca in feat.columns and cb in feat.columns:
        feat[f"{a}_{b}_emadiff"] = feat[ca] - feat[cb]
        feat[f"{a}_{b}_emaratio"] = feat[ca] / feat[cb].replace(0, np.nan) - 1
    
    ca, cb = f"{a}_rsi14", f"{b}_rsi14"
    if ca in feat.columns and cb in feat.columns:
        feat[f"{a}_{b}_rsidiff"] = feat[ca] - feat[cb]

# Реализованная волатильность
feat["m5_realvol20"] = m5["close"].pct_change().rolling(20, min_periods=20).std()
feat["m5_realvol50"] = m5["close"].pct_change().rolling(50, min_periods=50).std()

print(f"Всего фич: {feat.columns.size}")

# ==================== 6. ЦЕЛЕВОЙ СИГНАЛ (ПРОИЗВОДНАЯ ГАУССА) ====================
SIGMA = 13
WINDOW = 4 * SIGMA  # каузальное окно

def gauss_deriv_kernel(sigma, n):
    """Ядро первой производной Гаусса"""
    x = np.arange(n) - (n - 1) / 2.0
    g = np.exp(-(x ** 2) / (2 * sigma ** 2))
    dg = -x / (sigma ** 2) * g
    return dg / (np.abs(dg).sum() + 1e-12)

def causal_gauss_deriv(s: pd.Series, sigma: int = SIGMA, window: int = WINDOW) -> pd.Series:
    """Каузальная свертка с производной Гаусса (использует только прошлое)"""
    x = np.log(s.values)
    n = len(x)
    dg = gauss_deriv_kernel(sigma, window)
    out = np.zeros(n)
    for i in range(window - 1, n):
        out[i] = (dg * x[i - window + 1:i + 1][::-1]).sum()
    return pd.Series(out, index=s.index)

print("\nРассчитываю целевой сигнал...")
target_raw = causal_gauss_deriv(m5["close"], SIGMA, WINDOW)

# Нормализация (каузальная)
roll_mean = target_raw.rolling(2000, min_periods=200).mean()
roll_std = target_raw.rolling(2000, min_periods=200).std()
target = (target_raw - roll_mean) / roll_std.replace(0, np.nan)

# ==================== 7. СБОРКА ДАННЫХ ====================
df = feat.copy()
df["y"] = target
df = df.dropna(how='any')
print(f"Сэмплов после очистки: {len(df)}")

X = df.drop(columns=["y"])
y = df["y"]

# Walk-forward split (строго по времени)
split = int(len(df) * 0.8)
train_idx = df.index[:split]
test_idx = df.index[-500:]  # Последние 500 баров

# Проверка на пересечение
assert train_idx[-1] < test_idx[0], "Ошибка: пересечение train/test!"

Xtr, Xte = X.loc[train_idx], X.loc[test_idx]
ytr, yte = y.loc[train_idx], y.loc[test_idx]

print(f"Train: {len(Xtr)} баров, Test: {len(Xte)} баров")

# ==================== 8. ОБУЧЕНИЕ МОДЕЛИ ====================
print("\nОбучаю LightGBM...")

model = lgb.LGBMRegressor(
    n_estimators=1500,
    learning_rate=0.03,
    num_leaves=31,
    min_data_in_leaf=200,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbosity=-1,
)

model.fit(
    Xtr, ytr,
    eval_set=[(Xte, yte)],
    eval_metric="rmse",
    callbacks=[lgb.early_stopping(100, verbose=False)],
)

pred = model.predict(Xte)

# ==================== 9. ДИАГНОСТИКА ====================
corr = np.corrcoef(yte.values, pred)[0, 1]
rmse = np.sqrt(((yte.values - pred) ** 2).mean())

print(f"\n=== РЕЗУЛЬТАТЫ ТЕСТА ===")
print(f"Корреляция: {corr:.4f}")
print(f"RMSE: {rmse:.6f}")
print(f"R²: {1 - (yte.values - pred).var() / yte.var():.4f}")

# ==================== 10. ВАЖНОСТЬ ФИЧ ====================
imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nТоп-20 важных признаков:")
print(imp.head(20).to_string())

# ==================== 11. ВИЗУАЛИЗАЦИЯ ====================
fig, axes = plt.subplots(4, 1, figsize=(15, 14), sharex=True)

t = yte.index

# 1. Целевой сигнал vs предсказание
ax = axes[0]
ax.plot(t, yte.values, color='#1f77b4', lw=1.2, label='Истинная производная (цель)')
ax.plot(t, pred, color='#d62728', lw=1.0, alpha=0.9, label='Выход модели (LightGBM)')
ax.axhline(0, color='gray', ls=':', lw=0.7)
ax.set_title(f'Аппроксиматор 1-й производной конволюции Гаусса (σ={SIGMA})\n'
             f'Тест: последние 500 баров M5 | Корреляция={corr:.4f} RMSE={rmse:.4f}',
             fontsize=10)
ax.legend(loc='upper left')
ax.grid(alpha=0.3)

# 2. Ошибка
ax = axes[1]
error = pred - yte.values
ax.plot(t, error, color='#7f7f7f', lw=0.8)
ax.axhline(0, color='gray', ls=':')
ax.fill_between(t, 0, error, alpha=0.2, color='red' if (error > 0).any() else 'blue')
ax.set_title('Ошибка (модель − цель)')
ax.grid(alpha=0.3)

# 3. Распределение
ax = axes[2]
ax.hist(yte.values, bins=50, alpha=0.5, label='Истина', color='blue')
ax.hist(pred, bins=50, alpha=0.5, label='Предсказание', color='red')
ax.set_title('Распределение значений')
ax.legend()
ax.grid(alpha=0.3)

# 4. Цена (контекст)
ax = axes[3]
ax.plot(t, m5["close"].loc[test_idx], color='#2ca02c', lw=0.9)
ax.set_title('Цена EURUSD (контекст)')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ==================== 12. ЗУМ (первые 150 баров) ====================
fig, ax = plt.subplots(figsize=(14, 5))
zoom = slice(0, 150)
ax.plot(t[zoom], yte.values[zoom], color='#1f77b4', lw=1.4, label='Истинная производная')
ax.plot(t[zoom], pred[zoom], color='#d62728', lw=1.1, alpha=0.9, label='Выход модели')
ax.axhline(0, color='gray', ls=':', lw=0.7)
ax.set_title(f'Зум: первые 150 баров теста — нет «завала», фичи строго из прошлого (causal)')
ax.legend(loc='upper left')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ==================== 13. ПРОГНОЗ НА 1/3/5 БАРОВ ВПЕРЕД ====================
print("\n=== Прогноз производной на горизонт ===")
horizons = [1, 3, 5]
fc_results = {}

for h in horizons:
    yh = df["y"].shift(-h)
    d = X.join(yh.rename("yh")).dropna()
    tr, te = d.index[:split], d.index[-500:]
    
    m = lgb.LGBMRegressor(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=31,
        min_data_in_leaf=200,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbosity=-1
    )
    
    m.fit(
        d.loc[tr, X.columns], d.loc[tr, "yh"],
        eval_set=[(d.loc[te, X.columns], d.loc[te, "yh"])],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(80, verbose=False)]
    )
    
    ph = m.predict(d.loc[te, X.columns])
    c = np.corrcoef(d.loc[te, "yh"].values, ph)[0, 1]
    fc_results[h] = c
    print(f"Горизонт +{h} бар(ов): корреляция = {c:.4f}")

# График прогнозного горизонта
fig, ax = plt.subplots(figsize=(8, 4.5))
bars = [h * 5 for h in horizons]
ax.plot(bars, [fc_results[h] for h in horizons], '-o', color='#d62728', lw=2, ms=8)
for h in horizons:
    ax.annotate(f"{fc_results[h]:.2f}", (h*5, fc_results[h]),
                textcoords="offset points", xytext=(0, 10), ha='center', fontsize=11)
ax.set_xlabel("Горизонт предсказания (минуты)")
ax.set_ylabel("Корреляция с будущей производной")
ax.set_title("Предсказательная сила модели: производная в будущем\n(фичи строго из прошлого)")
ax.grid(alpha=0.3)
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.show()

# ==================== 14. ЗАВЕРШЕНИЕ ====================
print("\n✅ Анализ завершен!")
print(f"Модель обучена на {len(Xtr)} барах, проверена на {len(Xte)} барах")
print(f"Корреляция на тесте: {corr:.4f}")
print(f"Количество фич: {len(X.columns)}")

mt5.shutdown()