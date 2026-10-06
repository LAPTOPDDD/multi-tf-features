import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# --- 1. ЗАГРУЗКА ДАННЫХ ---
# Формат MT5: <DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>
df = pd.read_csv('EURUSD_M5_202504180935_202608271920.csv', sep='\t')

# Склеиваем дату и время в индекс
df['DateTime'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'])
df.set_index('DateTime', inplace=True)
df.sort_index(inplace=True)

# Приводим колонки к стандартному виду
data = df[['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<TICKVOL>']].copy()
data.columns = ['open', 'high', 'low', 'close', 'volume']

# --- 2. FEATURE ENGINEERING ---
# ATR (волатильность)
tr = np.maximum(data['high'] - data['low'], 
                np.maximum(abs(data['high'] - data['close'].shift(1)), 
                           abs(data['low'] - data['close'].shift(1))))
data['atr'] = tr.rolling(14).mean()

# Gaussian Derivative (твой любимый фильтр шума)
sigma = 5
window = sigma * 2 + 1
gauss_kernel = np.exp(-0.5 * ((np.arange(-sigma, sigma+1)/sigma)**2)) / (sigma * np.sqrt(2*np.pi))
smoothed_close = data['close'].rolling(window, center=True).apply(lambda x: np.dot(x, gauss_kernel), raw=True)
data['gauss_deriv'] = smoothed_close.diff()

# Объемная активность
data['vol_ratio'] = data['volume'] / data['volume'].rolling(50).mean()

# Канал Дончиана (база для пробоя)
data['donchian_high'] = data['high'].rolling(20).max()
data['donchian_low'] = data['low'].rolling(20).min()

# --- 3. ЦЕЛЕВАЯ ПЕРЕМЕННАЯ (TARGET) ---
# Условие успеха: цена ушла в нашу сторону > 1.5 ATR, не задев стоп < 1.0 ATR за N баров
lookahead = 10
tp_mult = 1.5
sl_mult = 1.0

future_high = data['high'].shift(-1).rolling(lookahead).max().shift(1) # shift(1) чтобы исключить текущий бар
future_low = data['low'].shift(-1).rolling(lookahead).min().shift(1)

# Бинарные таргеты
data['target_long'] = ((future_high - data['close']) > (data['atr'] * tp_mult)).astype(int)
data['target_short'] = ((data['close'] - future_low) > (data['atr'] * tp_mult)).astype(int)

# Объединяем: 1=Long, -1=Short, 0=Hold
data['signal'] = 0
data.loc[data['target_long'] == 1, 'signal'] = 1
data.loc[data['target_short'] == 1, 'signal'] = -1

# Удаляем NaN от индикаторов и lookahead
data.dropna(inplace=True)

# --- 4. ОБУЧЕНИЕ LIGHTGBM ---
features = ['atr', 'gauss_deriv', 'vol_ratio', 'donchian_high', 'donchian_low', 'close']
X = data[features]
y = data['signal']

# Разделение: последние 20% - строгий тест
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Обучение мультиклассовой модели
model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, verbose=-1, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
test_data = data.iloc[split_idx:].copy()
test_data['prediction'] = preds

# --- 5. СИМУЛЯЦИЯ ТОРГОВЛИ ---
capital = 10000
equity = [capital]
position = 0
entry_price = 0
current_atr = 0

for i in range(len(test_data)):
    row = test_data.iloc[i]
    price = row['close']
    atr = row['atr']
    
    if position == 0 and row['prediction'] != 0:
        # Вход по сигналу ML
        position = row['prediction']
        entry_price = price
        current_atr = atr
        
    elif position != 0:
        # Проверка выхода
        pnl_pct = (price - entry_price) / entry_price if position == 1 else (entry_price - price) / entry_price
        tp_level = current_atr * 2 / entry_price
        sl_level = current_atr / entry_price
        
        if pnl_pct >= tp_level or pnl_pct <= -sl_level:
            capital += capital * pnl_pct
            position = 0
            
    equity.append(capital)

# --- 6. ВИЗУАЛИЗАЦИЯ ---
plt.figure(figsize=(14, 6))
plt.plot(equity, color='green', linewidth=1.5, label='LGBM Enhanced Breakout Equity')
plt.title(f'Walk-Forward Test (Aug 2026) | Final: ${capital:,.2f} ({((capital-10000)/10000)*100:.1f}%)')
plt.xlabel('Bars (M5)')
plt.ylabel('Capital ($)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

print(f"✅ Тест завершен. Итоговый капитал: ${capital:,.2f}")
print(f"📊 Количество сделок: {len([e for e in equity if e != equity[-1]])}")