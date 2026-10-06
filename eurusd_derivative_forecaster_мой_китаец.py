#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EURUSD Derivative Forecaster (MT5 LIVE + offline mode)
=====================================================
ОДИН скрипт:
  1) берёт данные из запущенного и залогиненного MetaTrader 5 (EURUSD, M5)
     - либо, при --offline, читает CSV-файлы, лежащие рядом
  2) строит строго КАУЗАЛЬНЫЕ фичи со ВСЕХ таймфреймов (M5/M15/M30/H1/H4/Daily)
     - никаких утечек из будущего (rolling + EMA adjust=False + shift(1) при склейке ТФ)
  3) целевой сигнал = первая производная конволюции Гаусса (causal-свёртка) от log(close)
  4) обучает LightGBM-аппроксиматор этой производной
  5) на последних 500 барах M5 рисует: синюю цель vs красный выход модели

Запуск LIVE (MT5 должен быть открыт и в него нужно быть залогиненным):
    python eurusd_derivative_live.py

Тест на локальных CSV (без MT5):
    python eurusd_derivative_live.py --offline

Зависимости:
    pip install MetaTrader5 lightgbm pandas numpy matplotlib
"""

import os
import sys
import math
import argparse
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # чтобы работало без GUI, PNG сохраняется в файл
import matplotlib.pyplot as plt

try:
    import lightgbm as lgb
    HAVE_LGB = True
except ImportError:
    HAVE_LGB = False

try:
    import MetaTrader5 as mt5
    HAVE_MT5 = True
except ImportError:
    HAVE_MT5 = False


# =====================================================================
# ПАРАМЕТРЫ (меняй здесь)
# =====================================================================
SYMBOL = "EURUSD"
BASE_TF = "M5"                 # базовый таймфрейм (5 минут)
TIMEFRAMES = ["M5", "M15", "M30", "H1", "H4", "Daily"]
HISTORY_DAYS = 365              # сколько дней истории брать из MT5
SIGMA_BARS = 13                 # ширина ядра Гаусса (в барах M5, ~1 час)
TRAIN_FRAC = 0.80               # доля обучения (по времени, без перемешивания)
TEST_BARS = 500                 # сколько последних баров M5 — тест
N_ESTIMATORS = 1500
SEED = 42

# маппинг имён таймфреймов MT5
TF_MAP = {
    "M5": mt5.TIMEFRAME_M5 if HAVE_MT5 else None,
    "M15": mt5.TIMEFRAME_M15 if HAVE_MT5 else None,
    "M30": mt5.TIMEFRAME_M30 if HAVE_MT5 else None,
    "H1": mt5.TIMEFRAME_H1 if HAVE_MT5 else None,
    "H4": mt5.TIMEFRAME_H4 if HAVE_MT5 else None,
    "Daily": mt5.TIMEFRAME_D1 if HAVE_MT5 else None,
}
# минуты в одном баре каждого таймфрейма (для shift при склейке)
TF_MINUTES = {"M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "Daily": 1440}

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "derivative_out")
os.makedirs(OUTDIR, exist_ok=True)


def ensure_lightgbm():
    """Пытается импортировать / установить lightgbm + MetaTrader5 при необходимости."""
    global HAVE_LGB, HAVE_MT5
    if not HAVE_LGB:
        print("[INFO] lightgbm не найден — пытаюсь установить (pip install lightgbm)...")
        os.system(f"{sys.executable} -m pip install --quiet lightgbm")
        try:
            import lightgbm as lgb_tmp
            globals()["lgb"] = lgb_tmp
            HAVE_LGB = True
            print("[INFO] lightgbm установлен успешно.")
        except Exception as e:
            print(f"[WARN] не удалось установить lightgbm ({e}) — будет использован Ridge (хуже качество).")


def ensure_mt5():
    global HAVE_MT5
    if not HAVE_MT5:
        print("[INFO] MetaTrader5 не найден — пытаюсь установить...")
        os.system(f"{sys.executable} -m pip install --quiet MetaTrader5")
        try:
            import MetaTrader5 as mt5_tmp
            globals()["mt5"] = mt5_tmp
            HAVE_MT5 = True
        except Exception as e:
            print(f"[WARN] MetaTrader5 не удалось установить ({e}).")


# =====================================================================
# 1) ЗАГРУЗКА ДАННЫХ
# =====================================================================
def load_from_mt5(symbol, days):
    """Скачивает историю по всем таймфреймам из живого MT5."""
    if not HAVE_MT5:
        raise RuntimeError("MetaTrader5 не установлен (pip install MetaTrader5)")
    if not mt5.initialize():
        raise RuntimeError(
            "mt5.initialize() не удался. Убедись, что MetaTrader 5 ЗАПУЩЕН и ты В НЁМ ЗАЛОГИНЕН."
        )
    print(f"[MT5] terminal info: {mt5.terminal_info()}")
    print(f"[MT5] account: {mt5.account_info().login}")

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    data = {}
    for tf_name, tf_const in TF_MAP.items():
        rates = mt5.copy_rates_range(symbol, tf_const, start, end)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"MT5 вернул пусто для {symbol} {tf_name}")
        df = pd.DataFrame(rates)
        # MT5: time в секундах UTC
        df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df[["datetime", "open", "high", "low", "close", "tick_volume"]].rename(
            columns={"tick_volume": "tickvol"}
        ).sort_values("datetime").reset_index(drop=True)
        data[tf_name] = df
        print(f"[MT5] {tf_name}: {len(df)} баров  ({df['datetime'].iloc[0]} -> {df['datetime'].iloc[-1]})")

    mt5.shutdown()
    return data


def _find_header_row(path):
    """Ищет строку, содержащую '<DATE>' — она и есть заголовок (файлы MT5)."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            cells = line.strip().split()
            joined = " ".join(cells)
            if "<DATE>" in joined:
                return i
    return 0  # обычный CSV с заголовком в первой строке


def load_from_csv(folder):
    """
    Оффлайн-режим: читает EURUSD_<TF>_*.csv из папки.
    Поддерживает формат MT5, где в начале файла идут строки-метаданные
    вида '<DATE>', '<TIME>' и т.д., а затем — сами данные.
    """
    data = {}
    canonical = {
        "date": "<DATE>", "time": "<TIME>",
        "open": "<OPEN>", "high": "<HIGH>", "low": "<LOW>",
        "close": "<CLOSE>", "tickvol": "<TICKVOL>",
    }
    for tf_name, pat in {
        "M5": "EURUSD_M5", "M15": "EURUSD_M15", "M30": "EURUSD_M30",
        "H1": "EURUSD_H1", "H4": "EURUSD_H4", "Daily": "EURUSD_Daily",
    }.items():
        cand = [f for f in os.listdir(folder) if f.startswith(pat) and f.endswith(".csv")]
        if not cand:
            raise RuntimeError(f"В папке {folder} нет файла, начинающегося с {pat}")
        path = os.path.join(folder, sorted(cand)[-1])
        header = _find_header_row(path)
        # sep=None + engine='python' -> pandas сам угадывает разделитель (таб / запятая)
        df = pd.read_csv(path, skiprows=header, sep=None, engine="python")
        # на случай, если шапка склеилась в одну ячейку ('<DATE> <TIME> ...')
        if len(df.columns) == 1 and header == 0:
            raw = pd.read_csv(path, skiprows=header, sep=r"\s+", engine="python")
            if len(raw.columns) > 1:
                df = raw
        df = df.rename(columns={c: c.strip() for c in df.columns})

        # выбираем нужные колонки (работаем с обоими вариантами имён)
        sel = {}
        for key, alt in canonical.items():
            if alt in df.columns:
                sel[key] = alt
            elif key in df.columns:
                sel[key] = key
        # date/time/open/high/low/close обязательны, кроме time (у Daily может не быть)
        missing = [k for k in ("date", "open", "high", "low", "close") if k not in sel]
        if missing:
            raise RuntimeError(f"{pat}: не найдены колонки {missing}. Есть: {list(df.columns)}")
        if "time" not in sel:
            sel["time"] = None  # отметим, что времени нет

        needed = ["date", "time", "open", "high", "low", "close"]
        src = [sel[k] for k in needed if sel[k] is not None]
        df = df[src].copy()
        df.columns = [k for k in needed if sel[k] is not None]

        # только строки с настоящими датами
        df = df[df["date"].astype(str).str.match(r"\d{4}\.\d{2}\.\d{2}")].copy()
        if "time" in df.columns and df["time"].notna().any():
            df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"].astype(str), errors="coerce")
        else:
            # дневные бары (и выше) могут не иметь времени — берём дату как есть
            df["datetime"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["datetime"])
        for c in ["open", "high", "low", "close", "tickvol"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["open", "high", "low", "close"])
        if "tickvol" not in df.columns:
            df["tickvol"] = 0
        df = df[["datetime", "open", "high", "low", "close", "tickvol"]].sort_values("datetime").reset_index(drop=True)
        data[tf_name] = df
        print(f"[CSV] {tf_name}: {len(df)} баров  ({df['datetime'].iloc[0]} -> {df['datetime'].iloc[-1]})")
    return data


# =====================================================================
# 2) КАУЗАЛЬНЫЕ ИНДИКАТОРЫ (строго по прошлому, никаких утечек)
# =====================================================================
def add_indicators_causal(df, prefix):
    """
    Добавляет фичи к DataFrame df (таймфрейма prefix).
    Все окна — каузальные (rolling min_periods=window, EMA adjust=False).
    Возвращает ТОЛЬКО новые колонки (без сырых OHLC), чтобы потом их склеить.
    """
    c = df["close"]
    feats = pd.DataFrame(index=df.index)

    # ---- трендовые (EMA / SMA) ---------------------------------------
    for w in [5, 10, 20, 50]:
        feats[f"{prefix}_ema{w}"] = c.ewm(span=w, adjust=False).mean()
        feats[f"{prefix}_sma{w}"] = c.rolling(w, min_periods=w).mean()

    # ---- momentum ----------------------------------------------------
    for w in [5, 10, 20]:
        feats[f"{prefix}_mom{w}"] = c - c.shift(w)

    # ---- RSI (Wilder, causal) -----------------------------------------
    delta = c.diff()
    # векторизованный Wilder RSI (EMA от прибылей/убытков, adjust=False => строго causal)
    gain14 = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss14 = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain14 / loss14.replace(0, np.nan)
    feats[f"{prefix}_rsi14"] = 100 - 100 / (1 + rs)

    # ---- ATR (Welles Wilder, causal) ---------------------------------
    hl = df["high"] - df["low"]
    hc = (df["high"] - c.shift(1)).abs()
    lc = (df["low"] - c.shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    feats[f"{prefix}_atr14"] = tr.ewm(alpha=1/14, adjust=False).mean()

    # ---- Bollinger %B (causal) ---------------------------------------
    mid = c.rolling(20, min_periods=20).mean()
    sd = c.rolling(20, min_periods=20).std()
    feats[f"{prefix}_bbpctb"] = (c - mid) / (2 * sd)

    # ---- MACD (12/26/9, всё causal) ----------------------------------
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    feats[f"{prefix}_macd"] = macd
    feats[f"{prefix}_macdsig"] = macd.ewm(span=9, adjust=False).mean()
    feats[f"{prefix}_macdhist"] = feats[f"{prefix}_macd"] - feats[f"{prefix}_macdsig"]

    # ---- волатильность / оборот --------------------------------------
    feats[f"{prefix}_ret"] = c.pct_change()
    feats[f"{prefix}_vol20"] = feats[f"{prefix}_ret"].rolling(20, min_periods=20).std()
    feats[f"{prefix}_volatil"] = (c / c.shift(1) - 1).rolling(20, min_periods=20).std()
    feats[f"{prefix}_hl_range"] = (df["high"] - df["low"]) / c
    feats[f"{prefix}_logret"] = np.log(c / c.shift(1))

    # ---- trend slope (наклон EMA20) ---------------------------------
    feats[f"{prefix}_trend20"] = feats[f"{prefix}_ema20"].diff(5)

    # ffill только ВНУТРИ индикатора (NaN в начале окна — нормально)
    feats = feats.ffill().fillna(0.0)
    return feats


def build_feature_matrix(data):
    """
    Собирает матрицу фичей для базового таймфрейма (M5).
    Старшие таймфреймы привязываются через .shift(1), чтобы бар M5
    видел только УЖЕ ЗАКРЫВШИЕСЯ бары старшего ТФ (никакого look-ahead).
    """
    base = BASE_TF
    base_df = data[base].copy()
    base_df = base_df.reset_index(drop=True)

    feat_frames = []
    for tf in TIMEFRAMES:
        tf_df = data[tf].copy()
        # признак "бар ещё открыт / закрыт" нам не нужен — работаем по close
        f = add_indicators_causal(tf_df, prefix=tf.lower())

        # привязываем к сетке базового таймфрейма (asof + shift)
        f = f.reset_index().rename(columns={"index": "tf_idx"})
        # метка времени для merge_asof
        f["datetime"] = tf_df["datetime"].values
        feat_frames.append((tf, f))

    # собираем итоговую таблицу, проходя по базовым барам
    base_dt = base_df[["datetime"]].reset_index().rename(columns={"index": "base_idx"})
    all_feats = []
    for tf, f in feat_frames:
        m = pd.merge_asof(
            base_dt[["datetime", "base_idx"]],
            f[["datetime"] + [col for col in f.columns if col.startswith(tf.lower() + "_")]],
            on="datetime", direction="backward",
        )
        # <<< КЛЮЧЕВОЙ МОМЕНТ: сдвигаем фичи старшего ТФ на 1 базовый бар,
        #     чтобы исключить look-ahead (бар M5 не видит текущий незакрытый бар ТФ) >>>
        if tf != BASE_TF:
            cols = [c for c in m.columns if c.startswith(tf.lower() + "_")]
            m[cols] = m[cols].shift(1)
        m = m.set_index("base_idx")
        all_feats.append(m.drop(columns="datetime"))

    # один concat вместо множества .join (быстрее, без fragmentation)
    all_feats_df = pd.concat(all_feats, axis=1)

    # базовые цены тоже нужны (для цели и нормализации)
    out = pd.concat([base_df, all_feats_df.reset_index(drop=False)], axis=1)
    out = out.sort_values("datetime").reset_index(drop=True).copy()

    # ---- кросс-таймфреймные разности трендов (уже causal) ------------
    out["m5_m15_rsidiff"] = out["m5_rsi14"] - out["m15_rsi14"]
    out["m5_m15_emadiff"] = out["m5_ema20"] - out["m15_ema20"]
    out["h1_daily_emadiff"] = out["h1_ema20"] - out["daily_ema20"]

    # ---- заполнение NaN (только ffill вперёд по времени, без look-ahead) --
    skip = {"datetime", "open", "high", "low", "close", "tickvol", "base_idx"}
    raw_cols = [c for c in out.columns if c not in skip and out[c].dtype != object]
    out[raw_cols] = out[raw_cols].ffill().fillna(0.0)

    feat_cols = list(out.columns)
    feat_cols = [c for c in feat_cols if c not in skip and out[c].dtype != object]
    print(f"[FEATS] итого каузальных фичей: {len(feat_cols)}")
    return out, feat_cols


# =====================================================================
# 3) ЦЕЛЕВОЙ СИГНАЛ: первая производная конволюции Гаусса (causal)
# =====================================================================
def gaussian_derivative_target(log_close, sigma, window=None):
    """
    y_i = sum_{k=0}^{W-1} dG(k; sigma) * log_close[i - k]
    строго каузальная свёртка (ядро смотрит только в прошлое).
    dG/dx = -(x/sigma^3) * exp(-x^2/(2 sigma^2))
    """
    if window is None:
        window = int(sigma * 6) + 1
    x = np.arange(window)
    g = -(x / (sigma ** 3)) * np.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / (np.abs(g).sum() + 1e-12)  # нормировка
    # корреляция = свёртка с перевёрнутым ядром (ядро уже в нужном порядке)
    target = np.correlate(log_close, g, mode="full")[:len(log_close)]
    # начальные window баров — NaN (неполное окно)
    target[:window] = np.nan
    return target


# =====================================================================
# 4) ОБУЧЕНИЕ + ГРАФИК
# =====================================================================
def run(data, outdir):
    df, feat_cols = build_feature_matrix(data)

    log_close = np.log(df["close"].values)
    target = gaussian_derivative_target(log_close, sigma=SIGMA_BARS)
    df["target"] = target

    # убираем строки с NaN в цели (начало + первые window баров)
    valid = df["target"].notna()
    df = df[valid].reset_index(drop=True)
    print(f"[DATA] валидных баров после causal-свёртки: {len(df)}")

    n = len(df)
    n_train = int(n * TRAIN_FRAC)
    train = df.iloc[:n_train]
    test = df.iloc[n_train:]

    # берём ровно последние TEST_BARS баров теста (как ты просил — 500)
    if len(test) > TEST_BARS:
        test = test.iloc[-TEST_BARS:]
    print(f"[SPLIT] train={len(train)}  test={len(test)}")

    # ---- z-score нормировка, fitted ТОЛЬКО на train (никакой утечки из теста) ----
    train_mean = train[feat_cols].mean()
    train_std = train[feat_cols].std().replace(0, np.nan)
    # цель тоже нормируем по train, чтобы шкала фичей и цели совпадала
    y_mean = train["target"].mean()
    y_std = train["target"].std()

    def norm(df):
        x = (df[feat_cols] - train_mean) / train_std
        return x.fillna(0.0).values

    X_train = norm(train)
    X_test = norm(test)
    y_train = ((train["target"] - y_mean) / y_std).values
    y_test = ((test["target"] - y_mean) / y_std).values

    if not HAVE_LGB:
        print("[WARN] lightgbm не установлен — ставлю pip install lightgbm и использую Ridge")
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    else:
        model = lgb.LGBMRegressor(
            n_estimators=N_ESTIMATORS, learning_rate=0.02,
            num_leaves=127, min_child_samples=5,
            subsample=0.9, colsample_bytree=0.9,
            reg_alpha=0.0, reg_lambda=0.0,
            force_col_wise=True,
            random_state=SEED, n_jobs=-1,
        )
        model.fit(
            X_train, y_train,
            eval_X=X_test, eval_y=y_test,
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(300, verbose=False)],
        )
        y_pred = model.predict(X_test, num_iteration=model.best_iteration_)

    corr = np.corrcoef(y_test, y_pred)[0, 1]
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    print(f"[RESULT] corr(test) = {corr:.4f}   RMSE = {rmse:.4f}")

    # ---- важность фичей ------------------------------------------------
    if HAVE_LGB:
        imp = pd.DataFrame({"feat": feat_cols, "imp": model.feature_importances_})
    else:
        imp = pd.DataFrame({"feat": feat_cols, "imp": np.abs(model.coef_)})
    imp = imp.sort_values("imp", ascending=False).head(25)
    imp.to_csv(os.path.join(outdir, "top25_importance.csv"), index=False)
    print("[TOP5]\n" + imp.head(5).to_string(index=False))

    # ---- график -------------------------------------------------------
    ts = test["datetime"].values
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1, 2]})
    ax1, ax2, ax3 = axes

    ax1.plot(ts, y_test, color="#1f77b4", lw=1.2, label="цель (производная Гаусса)")
    ax1.plot(ts, y_pred, color="#d62728", lw=1.0, alpha=0.85, label="выход модели (LightGBM)")
    ax1.axhline(0, color="gray", lw=0.6)
    ax1.set_ylabel("производная (нормированная)")
    ax1.legend(loc="upper left")
    ax1.set_title(f"EURUSD {BASE_TF} — аппроксимация производной Гаусса\n"
                  f"corr = {corr:.3f}  |  RMSE = {rmse:.3f}  |  баров: {len(test)}",
                  fontsize=11)
    ax1.grid(alpha=0.3)

    ax2.bar(ts, y_test - y_pred, color="#ff7f0e", width=1.0)
    ax2.set_ylabel("ошибка")
    ax2.grid(alpha=0.3)

    ax3.plot(ts, df.loc[test.index, "close"].values if "close" in test.columns
             else test["close"].values, color="#2ca02c", lw=1.0)
    ax3.set_ylabel("EURUSD close")
    ax3.grid(alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()
    img_path = os.path.join(outdir, "derivative_forecast.png")
    fig.savefig(img_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {img_path}")

    # зум первых 150 баров теста (проверка, что нет "завала" на стыке)
    zoom = min(150, len(test))
    fig2, ax = plt.subplots(figsize=(15, 5))
    ax.plot(ts[:zoom], y_test[:zoom], color="#1f77b4", lw=1.5, label="цель")
    ax.plot(ts[:zoom], y_pred[:zoom], color="#d62728", lw=1.3, label="модель")
    ax.axhline(0, color="gray", lw=0.6)
    ax.legend(loc="upper left"); ax.grid(alpha=0.3)
    ax.set_title("Зум: начало теста (нет завала / провала)")
    fig2.tight_layout()
    zp = os.path.join(outdir, "derivative_zoom.png")
    fig2.savefig(zp, dpi=130, bbox_inches="tight")
    plt.close(fig2)
    print(f"[SAVED] {zp}")

    return corr, rmse


# =====================================================================
# MAIN
# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true",
                    help="читать CSV из папки (без MT5)")
    ap.add_argument("--folder", default="/data/inputs",
                    help="папка с CSV (только для --offline)")
    args = ap.parse_args()

    print("=" * 60)
    print("EURUSD Derivative Forecaster  |  causal features + LightGBM")
    print("=" * 60)

    ensure_lightgbm()

    if args.offline:
        print(f"[MODE] OFFLINE (CSV из {args.folder})")
        data = load_from_csv(args.folder)
    else:
        ensure_mt5()
        if not HAVE_MT5:
            print("[FATAL] MetaTrader5 недоступен. Установи: pip install MetaTrader5")
            print("        (нужен 64-битный Python и запущенный терминал MT5)")
            sys.exit(1)
        print(f"[MODE] LIVE (MT5: {SYMBOL}, {HISTORY_DAYS} дней)")
        data = load_from_mt5(SYMBOL, HISTORY_DAYS)

    corr, rmse = run(data, OUTDIR)
    print("-" * 60)
    print(f"ИТОГО: corr={corr:.4f}  RMSE={rmse:.4f}")
    print(f"Результаты в папке: {OUTDIR}")
    print("  -> derivative_forecast.png  (основной график: цель vs модель)")
    print("  -> derivative_zoom.png      (зум начала теста)")
    print("  -> top25_importance.csv     (важность фичей)")

    # открываем график в интерактивном окне (если есть GUI), иначе просто игнорируем
    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
