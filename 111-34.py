import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# --- 1. ЗАГРУЗКА ДАННЫХ (Универсальная) ---
FILE_PATH = 'EURUSD_H1_202401020000_202608271900.csv'

try:
    # Пробуем загрузить с заголовками
    df = pd.read_csv(FILE_PATH, sep='\t')
    
    # Проверяем, есть ли заголовки MT5
    if '<DATE>' in df.columns:
        print("✅ Обнаружен формат MT5 с заголовками")
        df['DateTime'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'])
        data = df[['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<TICKVOL>']].copy()
        data.columns = ['open', 'high', 'low', 'close', 'volume']
    else:
        # Формат без заголовков
        print("✅ Обнаружен формат без заголовков")
        df = pd.read_csv(FILE_PATH, sep='\t', header=None)
        df.columns = ['date', 'time', 'open', 'high', 'low', 'close', 'volume', 'vol', 'spread']
        df['DateTime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        data = df[['open', 'high', 'low', 'close', 'volume']].copy()
    
    # Устанавливаем индекс
    data.index = df['DateTime']
    data.sort_index(inplace=True)
    
    print(f"✅ Загружено {len(data)} баров. Период: {data.index[0]} - {data.index[-1]}")
    
except Exception as e:
    print(f"❌ Ошибка загрузки: {e}")
    exit()

# --- 2. ГЕНЕРАЦИЯ ФИЧЕЙ ---
# ATR
tr = np.maximum(data['high'] - data['low'], 
                np.maximum(abs(data['high'] - data['close'].shift(1)), 
                           abs(data['low'] - data['close'].shift(1))))
data['atr'] = tr.rolling(14).mean()

# Gaussian Derivative
sigma = 3
window = sigma * 2 + 1
gauss_kernel = np.exp(-0.5 * ((np.arange(-sigma, sigma+1)/sigma)**2)) / (sigma * np.sqrt(2*np.pi))
smoothed_close = data['close'].rolling(window, center=True).apply(lambda x: np.dot(x, gauss_kernel), raw=True)
data['gauss_deriv'] = smoothed_close.diff()

# Объем и канал Дончиана
data['vol_ratio'] = data['volume'] / data['volume'].rolling(50).mean()
data['donchian_high'] = data['high'].rolling(20).max()
data['donchian_low'] = data['low'].rolling(20).min()
data['hour'] = data.index.hour

# --- 3. ЦЕЛЕВАЯ ПЕРЕМЕННАЯ ---
lookahead = 12
tp_mult = 1.5
sl_mult = 1.0

future_high = data['high'].shift(-1).rolling(lookahead).max().shift(1)
future_low = data['low'].shift(-1).rolling(lookahead).min().shift(1)

data['target_long'] = ((future_high - data['close']) > (data['atr'] * tp_mult)).astype(int)
data['target_short'] = ((data['close'] - future_low) > (data['atr'] * tp_mult)).astype(int)

data['signal'] = 0
data.loc[data['target_long'] == 1, 'signal'] = 1
data.loc[data['target_short'] == 1, 'signal'] = -1

data.dropna(inplace=True)

# --- 4. WALK-FORWARD С ПЕРЕОБУЧЕНИЕМ ---
features = ['atr', 'gauss_deriv', 'vol_ratio', 'donchian_high', 'donchian_low', 'hour', 'close']

train_months = 6
test_months = 1
step_months = 1

dates = data.index
start_date = dates[0]
end_date = dates[-1]

train_windows = []
test_windows = []

current_start = start_date
while current_start + pd.DateOffset(months=train_months + test_months) <= end_date:
    train_end = current_start + pd.DateOffset(months=train_months)
    test_end = train_end + pd.DateOffset(months=test_months)
    
    train_windows.append((current_start, train_end))
    test_windows.append((train_end, test_end))
    
    current_start += pd.DateOffset(months=step_months)

capital = 10000
equity_curve = [capital]
trade_log = []

print(f"\n🔄 Запуск Walk-Forward: {len(train_windows)} окон")

for i, (tr_s, tr_e) in enumerate(train_windows):
    te_s, te_e = test_windows[i]
    
    train_data = data[(data.index >= tr_s) & (data.index < tr_e)]
    test_data = data[(data.index >= te_s) & (data.index < te_e)]
    
    if len(train_data) < 100 or len(test_data) < 10:
        continue
        
    X_train = train_data[features]
    y_train = train_data['signal']
    X_test = test_data[features]
    
    model = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, max_depth=4, verbose=-1, random_state=42)
    model.fit(X_train, y_train)
    
    probs = model.predict_proba(X_test)
    preds = np.argmax(probs, axis=1)
    
    position = 0
    entry_price = 0
    current_atr = 0
    
    for idx in range(len(test_data)):
        row = test_data.iloc[idx]
        price = row['close']
        atr = row['atr']
        
        pred_class = preds[idx]
        confidence = probs[idx][pred_class]
        
        if position == 0 and confidence > 0.60 and pred_class != 0:
            position = 1 if pred_class == 1 else -1
            entry_price = price
            current_atr = atr
            
        elif position != 0:
            pnl_pct = (price - entry_price) / entry_price if position == 1 else (entry_price - price) / entry_price
            tp_level = current_atr * 2 / entry_price
            sl_level = current_atr / entry_price
            
            if pnl_pct >= tp_level or pnl_pct <= -sl_level:
                capital += capital * pnl_pct
                trade_log.append({'date': row.name, 'pnl': pnl_pct})
                position = 0
                
        equity_curve.append(capital)
        
    print(f"✅ Окно {i+1}/{len(train_windows)} | Capital: ${capital:.2f}")

# --- 5. ВИЗУАЛИЗАЦИЯ ---
plt.figure(figsize=(14, 6))
plt.plot(equity_curve, color='green', linewidth=1.5, label='Walk-Forward Equity')
plt.axhline(y=10000, color='gray', linestyle='--', alpha=0.5)
plt.title(f'H1 EURUSD Walk-Forward | Final: ${capital:,.2f} ({((capital-10000)/10000)*100:.1f}%)')
plt.xlabel('Date')
plt.ylabel('Capital ($)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

print(f"\n📊 Итоговая доходность: {((capital-10000)/10000)*100:.2f}%")
if trade_log:
    win_rate = sum(1 for t in trade_log if t['pnl'] > 0) / len(trade_log)
    print(f"🎯 Win Rate: {win_rate*100:.1f}% | Сделок: {len(trade_log)}")