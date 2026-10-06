import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# --- 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ ---
# Используем M5 для более частых сигналов, так как у нас мало истории (2 года)
df = pd.read_csv('EURUSD_M5_202504180935_202608271920.csv', sep='\t') # Проверь разделитель, если ошибка - замени на ','

# Переименуем колонки для удобства (предполагаем стандартный MT5 экспорт)
# Если колонки без названий, раскомментируй строку ниже:
# df.columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'TickVol', 'Vol', 'Spread']

# Преобразуем дату
df['DateTime'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>']) if '<DATE>' in df.columns else pd.to_datetime(df.iloc[:, 0] + ' ' + df.iloc[:, 1])
df.set_index('DateTime', inplace=True)
df = df.sort_index()

# Берем только нужные колонки
data = df[['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<TICKVOL>']].copy()
data.columns = ['open', 'high', 'low', 'close', 'volume']

# --- 2. ГЕНЕРАЦИЯ ФИЧЕЙ (FEATURE ENGINEERING) ---

# 1. Волатильность (ATR)
data['tr'] = np.maximum(data['high'] - data['low'], 
                        np.maximum(abs(data['high'] - data['close'].shift(1)), 
                                   abs(data['low'] - data['close'].shift(1))))
data['atr'] = data['tr'].rolling(window=14).mean()

# 2. Производная Гаусса (твой любимый фильтр шума)
# Сглаживаем цену и берем разницу
sigma = 5
gaussian_prices = data['close'].rolling(window=sigma*2+1, center=True).apply(lambda x: np.dot(x, np.exp(-0.5 * ((np.arange(-sigma, sigma+1)/sigma)**2)) / (sigma * np.sqrt(2*np.pi))), raw=True)
data['gauss_deriv'] = gaussian_prices.diff()

# 3. Моментум и сила тренда
data['roc'] = data['close'].pct_change(periods=10)
data['vol_ratio'] = data['volume'] / data['volume'].rolling(50).mean()

# 4. Диапазон для пробоя (Donchian 20 периодов)
data['high_20'] = data['high'].rolling(20).max()
data['low_20'] = data['low'].rolling(20).min()

# --- 3. СОЗДАНИЕ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ (TARGET) ---
# Цель: Предсказать, будет ли цена выше закрытия + ATR через 10 баров?
lookahead = 10
tp_multiplier = 1.5
sl_multiplier = 1.0

data['future_high'] = data['high'].shift(-lookahead).rolling(lookahead).max() # Максимум в будущем окне
data['future_low'] = data['low'].shift(-lookahead).rolling(lookahead).min()   # Минимум в будущем окне

# Бинарный таргет: 1 если пробой вверх успешен, 0 иначе
# Условие: Цена ушла вверх больше чем на 1.5 ATR, не задев стопа вниз
data['target_up'] = ((data['future_high'] - data['close']) > (data['atr'] * tp_multiplier)).astype(int)
data['target_down'] = ((data['close'] - data['future_low']) > (data['atr'] * tp_multiplier)).astype(int)

# Объединяем: 1 - Long, -1 - Short, 0 - Hold
data['signal_raw'] = 0
data.loc[data['target_up'] == 1, 'signal_raw'] = 1
data.loc[data['target_down'] == 1, 'signal_raw'] = -1

# Удаляем NaN
data.dropna(inplace=True)

# --- 4. ОБУЧЕНИЕ LIGHTGBM ---
features = ['atr', 'gauss_deriv', 'roc', 'vol_ratio', 'high_20', 'low_20', 'close']
target_col = 'signal_raw'

# Разделяем на train/test (последние 20% данных - тест)
split = int(len(data) * 0.8)
train = data.iloc[:split]
test = data.iloc[split:]

X_train = train[features]
y_train = train[target_col]
X_test = test[features]
y_test = test[target_col]

# Модель для определения направления (Long/Short)
# Мы обучаем две модели или одну мультиклассовую. Для простоты возьмем мультикласс
model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, verbose=-1)
model.fit(X_train, y_train)

# Предсказания
preds = model.predict(X_test)
probs = model.predict_proba(X_test)

# --- 5. ТОРГОВАЯ СИМУЛЯЦИЯ И ЭКВИТИ ---
capital = 10000
equity = [capital]
position = 0 # 0 - нет позиции, 1 - long, -1 - short
entry_price = 0
atr_val = 0

# Проходим по тестовой выборке
test_data = test.copy()
test_data['prediction'] = preds

for i in range(len(test_data)):
    row = test_data.iloc[i]
    current_price = row['close']
    current_atr = row['atr']
    
    # Логика управления позицией
    if position == 0:
        # Если есть сигнал от ML
        if row['prediction'] == 1: # Long
            position = 1
            entry_price = current_price
            atr_val = current_atr
        elif row['prediction'] == -1: # Short
            position = -1
            entry_price = current_price
            atr_val = current_atr
            
    elif position != 0:
        # Проверка выхода по стопу или тейку (упрощенно: фиксированный R:R)
        pnl_pct = 0
        if position == 1:
            pnl_pct = (current_price - entry_price) / entry_price
            # Выход если прибыль > 2*ATR или убыток < -1*ATR
            if pnl_pct > (atr_val * 2 / entry_price) or pnl_pct < -(atr_val / entry_price):
                capital += capital * pnl_pct
                position = 0
        elif position == -1:
            pnl_pct = (entry_price - current_price) / entry_price
            if pnl_pct > (atr_val * 2 / entry_price) or pnl_pct < -(atr_val / entry_price):
                capital += capital * pnl_pct
                position = 0
                
    equity.append(capital)

# --- 6. ВИЗУАЛИЗАЦИЯ ---
plt.figure(figsize=(14, 7))
plt.plot(equity, label='Equity Curve (ML Enhanced)', color='green')
plt.title('LightGBM Enhanced Breakout Strategy Equity')
plt.xlabel('Bars')
plt.ylabel('Capital ($)')
plt.grid(True)
plt.legend()
plt.show()

print(f"Начальный капитал: $10,000")
print(f"Конечный капитал: ${capital:.2f}")
print(f"Доходность: {((capital - 10000) / 10000) * 100:.2f}%")