# This file calculates 5 techincal feature:
# Daily Variation: The difference between the High and Low columns, divided by the Open column. This feature represents the volatility of the index on that day.

# Daily Return: The percentage change in the Close column from the previous day's Close column. This feature represents the performance of the index on that day.

# 7-Day SMA: The 7-day simple moving average of the Close column. This feature represents the short-term trend of the index.

# 7-Day STD: The 7-day standard deviation of the Close column. This feature represents the short-term variability of the index.

# 14-Day EMA: The 14-day exponential moving average of the Close column. This feature represents a smoother and more responsive version of the SMA.

# MACD: The moving average convergence divergence, calculated from a 12-day EMA and a 26-day EMA of Close % Change. 
# This feature is another popular technical indicator that measures the trend and momentum of an asset.

import numpy as np
import pandas as pd

INPUT_CSV   = 'all_stocks_5yr.csv'
OUTPUT_CSV  = 'stocks_with_indicators.csv' # output csv

# === LOAD & PREP ===
df = pd.read_csv(INPUT_CSV, parse_dates=['date'])

grouped = df.groupby('Name', group_keys=False)

# DAILY VARIATION
# (High – Low) ÷ Open
df['daily_variation'] = (df['high'] - df['low']) / df['open']

# DAILY RETURN
# % change in Close from prior day
df['daily_return'] = grouped['close'].transform(lambda x: x.pct_change())

# 5-DAY SMA & STD (approximately 1 trading week)
df['sma_7'] = grouped['close'].transform(lambda x: x.rolling(window=5).mean())
df['std_7'] = grouped['close'].transform(lambda x: x.rolling(window=5).std())

# 10-DAY EMA (approximately 2 trading weeks)
df['ema_14'] = grouped['close'].transform(lambda x: x.ewm(span=10, adjust=False).mean())

# MACD ON %-CHANGE
ema12_ret = grouped['daily_return'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
ema26_ret = grouped['daily_return'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
df['macd']    = ema12_ret - ema26_ret

# MACD SIGNAL LINE
df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

"""===================Amelie's Section========================================"""
# CUMULATIVE RETURN
first_close = df['close'].iloc[0]
df['cumulative_return'] = df.groupby('Name').apply(lambda g: ((g['close'] - g['close'].iloc[0]) / g['close'].iloc[0]) * 100).reset_index(level=0, drop=True)

# Gain and Loss - Note: This might take up a lot of space so I'm not sure if you want to keep these columns or not
df['gain'] = df['daily_return'].apply(lambda x: x if x > 0 else 0)
df['loss'] = df['daily_return'].apply(lambda x: -x if x < 0 else 0)

# RSI
avg_gain = df.groupby('Name')['gain'].rolling(window=14, min_periods=1).mean().reset_index(level=0, drop=True)
avg_loss = df.groupby('Name')['loss'].rolling(window=14, min_periods=1).mean().reset_index(level=0, drop=True)
rs = avg_gain / avg_loss
df['rsi'] = 100 - (100 / (1 + rs))

# Stochastic Oscillator
df['L14'] = df.groupby('Name')['low'].rolling(window=14, min_periods=1).min().reset_index(level=0, drop=True)
df['H14'] = df.groupby('Name')['high'].rolling(window=14, min_periods=1).max().reset_index(level=0, drop=True)
df['stochastic_oscillator'] = ((df['close'] - df['L14']) / (df['H14'] - df['L14'])) * 100

# ATR (Average True Range, used to calculate volatility) over 14 days
# True Range (could be useful for other calcs)
# Shift the data to get previous day's values
df['prev_high'] = df.groupby('Name')['high'].shift(1)
df['prev_low'] = df.groupby('Name')['low'].shift(1)
df['prev_close'] = df.groupby('Name')['close'].shift(1)

tr1 = df['high'] - df['low']
tr2 = np.abs(df['high'] - df['prev_close'])
tr3 = np.abs(df['low'] - df['prev_close'])

df['true_range'] = np.maximum.reduce([tr1, tr2, tr3])

# ATR (Average True Range)
df['atr'] = df.groupby('Name')['true_range'].transform(lambda x: x.ewm(alpha=1/14, adjust=False).mean())

# ADX (Average Directional Index), used to calculate strength and direction of the index over 14 days
# Directional Movement (DM) calculations
df['plus_dir'] = df['high'] - df['prev_high']
df['minus_dir'] = df['prev_low'] - df['low']

# Only keep positive directional movement
df['plus_dm'] = np.where((df['plus_dir'] > df['minus_dir']) & (df['plus_dir'] > 0), df['plus_dir'], 0)
df['minus_dm'] = np.where((df['minus_dir'] > df['plus_dir']) & (df['minus_dir'] > 0), df['minus_dir'], 0)

# Smooth +DM and -DM over 14 periods per stock
df['smoothed_plus_dm'] = df.groupby('Name')['plus_dm'].transform(lambda x: x.ewm(span=14, adjust=False).mean())
df['smoothed_minus_dm'] = df.groupby('Name')['minus_dm'].transform(lambda x: x.ewm(span=14, adjust=False).mean())

df['smoothed_plus_dm'] = (df['smoothed_plus_dm'] / df['atr']) * 100
df['smoothed_minus_dm'] = (df['smoothed_minus_dm'] / df['atr']) * 100

# Calculate DX
df['dx'] = (np.abs(df['smoothed_plus_dm'] - df['smoothed_minus_dm']) / np.abs((df['smoothed_plus_dm'] + df['smoothed_minus_dm']))) * 100

# Calculate ADX (14-period smoothing of DX)
df['adx'] = df.groupby('Name')['dx'].transform(lambda x: x.ewm(span=14, adjust=False).mean())

# Actual price 3 and 7 days out
df['future_price_3'] = df.groupby('Name')['close'].shift(periods=-3)
df['future_price_7'] = df.groupby('Name')['close'].shift(periods=-5)

# 3 and 7 day BUY, HOLD, SELL tags
# Calculate future returns using trading days (approximately 5 trading days per week)
df['future_return_3'] = df.groupby('Name')['close'].transform(lambda x: x.shift(-3) / x - 1)  # 3 trading days
df['future_return_7'] = df.groupby('Name')['close'].transform(lambda x: x.shift(-5) / x - 1)  # 5 trading days (approximately 1 week)

# Use FUTURE return
def create_label(return_value):
    if return_value > 0.03:  # 3% threshold
        return 'buy'
    elif return_value < -0.03:  # -3% threshold
        return 'sell'
    else:
        return 'hold'

df['label_3'] = df['future_return_3'].apply(create_label)
df['label_7'] = df['future_return_7'].apply(create_label)

# Drop future return columns
df = df.drop(['future_return_3', 'future_return_7'], axis=1)

# Output to new CSV
df.to_csv(OUTPUT_CSV, index=False)
print(f"Wrote indicators to {OUTPUT_CSV}")
