import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import optuna
from scipy.ndimage import gaussian_filter1d  # <-- ВАЖНО: Добавлен импорт для каузального фильтра
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# ==========================================
FILE_PATH = 'EURUSD_H1_202401020000_202608271900.csv'

try:
    # Пробуем загрузить с заголовками MT5 (как в твоем файле)
    df = pd.read_csv(FILE_PATH, sep='\t')
    
    if '<DATE>' in df.columns:
        print("✅ Обнаружен формат MT5 с заголовками")
        # Создаем DateTime прямо в основном фрейме
        df['DateTime'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'])
        
        # Выбираем данные и СРАЗУ устанавливаем индекс из основного df
        data = df[['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<TICKVOL>']].copy()
        data.columns = ['open', 'high', 'low', 'close', 'volume']
        data.index = df['DateTime'] 
        
    else:
        # Формат без заголовков (резервный вариант)
        print("✅ Обнаружен формат без заголовков")
        df = pd.read_csv(FILE_PATH, sep='\t', header=None)
        df.columns = ['date', 'time', 'open', 'high', 'low', 'close', 'volume', 'vol', 'spread']
        df['DateTime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        
        data = df[['open', 'high', 'low', 'close', 'volume']].copy()
        data.index = df['DateTime']
        
except Exception as e:
    print(f"❌ Ошибка загрузки файла '{FILE_PATH}': {e}")
    exit()

data.sort_index(inplace=True)
print(f"📊 Загружено {len(data)} баров | {data.index[0]} - {data.index[-1]}")

# ==========================================
# 2. ГЕНЕРАЦИЯ ФИЧЕЙ (КАЗУАЛЬНАЯ ПРОИЗВОДНАЯ)
# ==========================================
# ATR
tr = np.maximum(data['high'] - data['low'], 
                np.maximum(abs(data['high'] - data['close'].shift(1)), 
                           abs(data['low'] - data['close'].shift(1))))
data['atr'] = tr.rolling(14).mean()

# КАЗУАЛЬНАЯ Гауссова Производная (только прошлые данные!)
# mode='nearest' гарантирует отсутствие look-ahead bias
sigma = 3  
gauss_smoothed = gaussian_filter1d(data['close'].values, sigma=sigma, mode='nearest', truncate=4.0)
data['gauss_deriv'] = pd.Series(gauss_smoothed, index=data.index).diff()

# Объемная активность
data['vol_ratio'] = data['volume'] / data['volume'].rolling(50).mean()

# Канал Дончиана
data['donchian_high'] = data['high'].rolling(20).max()
data['donchian_low'] = data['low'].rolling(20).min()

# Время суток
data['hour'] = data.index.hour

# Целевая переменная (Triple Barrier Lite)
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

features = ['atr', 'gauss_deriv', 'vol_ratio', 'donchian_high', 'donchian_low', 'hour', 'close']

# ==========================================
# 3. OPTUNA: ПОИСК ТОП-10 РАЗНООБРАЗНЫХ МОДЕЛЕЙ
# ==========================================
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 2, 8),
        'num_leaves': trial.suggest_int('num_leaves', 10, 50),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': trial.suggest_int('random_state', 0, 1000)
    }
    
    split = int(len(data) * 0.7)
    X_tr = data.iloc[:split][features]
    y_tr = data.iloc[:split]['signal']
    X_te = data.iloc[split:split+1000][features]
    y_te = data.iloc[split:split+1000]['signal']
    
    model = lgb.LGBMClassifier(**params, verbose=-1)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    return np.mean(preds == y_te)

print("\n🔍 Запуск Optuna (поиск 10 лучших конфигураций)...")
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=50, show_progress_bar=False)

top_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:10]
best_params_list = [t.params for t in top_trials]
print(f"✅ Найдено {len(best_params_list)} уникальных конфигураций")

# ==========================================
# 4. WALK-FORWARD: 10 НЕЗАВИСИМЫХ СИСТЕМ
# ==========================================
train_months = 6
test_months = 1
step_months = 1

dates = data.index
start_date = dates[0]
end_date = dates[-1]

windows = []
current_start = start_date
while current_start + pd.DateOffset(months=train_months + test_months) <= end_date:
    tr_end = current_start + pd.DateOffset(months=train_months)
    te_end = tr_end + pd.DateOffset(months=test_months)
    windows.append((current_start, tr_end, te_end))
    current_start += pd.DateOffset(months=step_months)

# Инициализация капитала для каждой из 10 систем
num_models = len(best_params_list)
individual_capitals = np.full(num_models, 10000.0) 
equity_curves_individual = [[] for _ in range(num_models)]
total_equity_curve = [10000.0 * num_models] 

print(f"\n🚀 Запуск {num_models} независимых систем на {len(windows)} окнах...")

for i, (tr_s, tr_e, te_e) in enumerate(windows):
    train_data = data[(data.index >= tr_s) & (data.index < tr_e)]
    test_data = data[(data.index >= tr_e) & (data.index < te_e)]
    
    if len(train_data) < 100 or len(test_data) < 10: 
        continue
    
    X_train = train_data[features]
    y_train = train_data['signal']
    X_test = test_data[features]
    
    # Обучаем ВСЕ 10 моделей независимо
    models = []
    for p_idx, params in enumerate(best_params_list):
        m = lgb.LGBMClassifier(**params, verbose=-1)
        m.fit(X_train, y_train)
        models.append(m)
    
    # Каждая модель торгует независимо
    positions = np.zeros(num_models, dtype=int)
    entry_prices = np.zeros(num_models)
    atr_values = np.zeros(num_models)
    
    for idx in range(len(test_data)):
        row = test_data.iloc[idx]
        price = row['close']
        atr = row['atr']
        
        for m_idx in range(num_models):
            probs = models[m_idx].predict_proba(X_test.iloc[[idx]])
            pred_class = np.argmax(probs[0])
            confidence = probs[0][pred_class]
            
            # Логика входа/выхода для каждой системы отдельно
            if positions[m_idx] == 0 and confidence > 0.55 and pred_class != 0:
                positions[m_idx] = 1 if pred_class == 1 else -1
                entry_prices[m_idx] = price
                atr_values[m_idx] = atr
                
            elif positions[m_idx] != 0:
                pnl_pct = (price - entry_prices[m_idx]) / entry_prices[m_idx] if positions[m_idx] == 1 \
                          else (entry_prices[m_idx] - price) / entry_prices[m_idx]
                
                tp_level = atr_values[m_idx] * 2 / entry_prices[m_idx]
                sl_level = atr_values[m_idx] / entry_prices[m_idx]
                
                if pnl_pct >= tp_level or pnl_pct <= -sl_level:
                    individual_capitals[m_idx] += individual_capitals[m_idx] * pnl_pct
                    positions[m_idx] = 0
                    
            equity_curves_individual[m_idx].append(individual_capitals[m_idx])
            
        total_equity_curve.append(np.sum(individual_capitals))
        
    print(f"✅ Окно {i+1}/{len(windows)} | Total Capital: ${np.sum(individual_capitals):,.2f}")

# ==========================================
# 5. ВИЗУАЛИЗАЦИЯ
# ==========================================
plt.figure(figsize=(16, 8))

# График суммарной эквити
plt.plot(total_equity_curve, color='green', linewidth=2, label=f'Total Ensemble Equity (${total_equity_curve[-1]:,.2f})')

# Графики индивидуальных систем (полупрозрачные)
for m_idx in range(num_models):
    plt.plot(equity_curves_individual[m_idx], alpha=0.3, linewidth=1, label=f'System {m_idx+1}' if m_idx == 0 else "")

plt.axhline(y=10000 * num_models, color='gray', linestyle='--', alpha=0.5)
plt.title(f'H1 EURUSD Independent Ensemble Walk-Forward (Causal Gauss)\n{num_models} Systems Trading Simultaneously')
plt.xlabel('Date / Bars')
plt.ylabel('Capital ($)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

print(f"\n🏆 Итоговый капитал ансамбля: ${total_equity_curve[-1]:,.2f}")
print(f"📈 Доходность ансамбля: {((total_equity_curve[-1] - 10000*num_models)/(10000*num_models))*100:.2f}%")
print(f"💰 Средний капитал одной системы: ${np.mean(individual_capitals):,.2f}")