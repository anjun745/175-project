# This file calculates 5 techincal feature:
# Daily Variation: The difference between the High and Low columns, divided by the Open column. This feature represents the volatility of the index on that day.

# Daily Return: The percentage change in the Close column from the previous day’s Close column. This feature represents the performance of the index on that day.

# 7-Day SMA: The 7-day simple moving average of the Close column. This feature represents the short-term trend of the index.

# 7-Day STD: The 7-day standard deviation of the Close column. This feature represents the short-term variability of the index.

# 14-Day EMA: The 14-day exponential moving average of the Close column. This feature represents a smoother and more responsive version of the SMA.

# MACD: The moving average convergence divergence, calculated from a 12-day EMA and a 26-day EMA of Close % Change. 
# This feature is another popular technical indicator that measures the trend and momentum of an asset.

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

# 7-DAY SMA & STD
df['sma_7'] = grouped['close'].transform(lambda x: x.rolling(window=7).mean())
df['std_7'] = grouped['close'].transform(lambda x: x.rolling(window=7).std())

# 14-DAY EMA
df['ema_14'] = grouped['close'].transform(lambda x: x.ewm(span=14, adjust=False).mean())

# MACD ON %-CHANGE
ema12_ret = grouped['daily_return'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
ema26_ret = grouped['daily_return'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
df['macd']    = ema12_ret - ema26_ret

# MACD SIGNAL LINE
df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

"""======Amelie's Section=========="""
# CUMULATIVE RETURN
"""first_close = df['close'].iloc[0]
df['cumulative_return'] = ((df['close'] - first_close) / first_close) * 100

# Gain and Loss - Note: This might take up a lot of space so I'm not sure if you want to keep these columns or not
df['gain'] = df['daily_return'].apply(lambda x: x if x > 0 else 0)
df['loss'] = df['daily_return'].apply(lambda x: -x if x < 0 else 0)

# RSI
avg_gain = df.groupby('Name')['gain'].rolling(window=14, min_periods=1).mean().reset_index(level=0, drop=True)
avg_loss = df.groupby('Name')['loss'].rolling(window=14, min_periods=1).mean().reset_index(level=0, drop=True)
rs = avg_gain / avg_loss
df['rsi'] = 100 - (100 / (1 + rs))"""

# Output to new CSV
df.to_csv(OUTPUT_CSV, index=False)
print(f"Wrote indicators to {OUTPUT_CSV}")
