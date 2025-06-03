import os
import pandas as pd
import numpy as np


csv_path = 'selected_data_with_nn.csv'
if not os.path.isfile(csv_path):
    raise SystemExit(f"Error: '{csv_path}' not found. Place it alongside this script.")

df = pd.read_csv(csv_path, parse_dates=['date'])


unique_stocks = df['stock'].unique()
num_stocks = len(unique_stocks)

# $1,000,000 equally split
portfolio_size = 1_000_000.0
stock_budget = portfolio_size / num_stocks

# Each trade is 10% of that stock’s budget
trade_size = 0.10 * stock_budget


out_dir = 'portfolio_details'
os.makedirs(out_dir, exist_ok=True)
trade_list_csv = os.path.join(out_dir, 'trade_list.csv')

records = []      
summary_rows = [] 

# Simulate trades

def simulate_and_record_exit_on_sell(model_col: str):
    for entry_type in ['open', 'close']:
        for exit_type in ['open', 'close']:
            strategy_name = f"{model_col}_entry_{entry_type}_exit_{exit_type}"
            total_profit = 0.0
            total_trades = 0

            for stock in unique_stocks:
                # Subset and sort this tickers data by date
                df_stock = df[df['stock'] == stock].sort_values('date').reset_index(drop=True)
                n = len(df_stock)

                for i in range(n):
                    row_i = df_stock.iloc[i]
                    signal_i = str(row_i.get(model_col, '')).strip().lower()
                    if signal_i != 'buy':
                        continue

                    # Must have a valid entry price on row i
                    if pd.isna(row_i[entry_type]):
                        continue
                    entry_price = float(row_i[entry_type])

                    # Determine how many shares can be bought
                    shares = int(np.floor(trade_size / entry_price))
                    if shares < 1:
                        continue

                    # Look for the next sell after day i
                    exit_index = None
                    for j in range(i + 1, n):
                        row_j = df_stock.iloc[j]
                        if str(row_j.get(model_col, '')).strip().lower() == 'sell':
                            # If the exit_type price is present, pick this as exit
                            if not pd.isna(row_j[exit_type]):
                                exit_index = j
                            break

                    if exit_index is not None:
                        # Found a valid sell at row j
                        exit_row = df_stock.iloc[exit_index]
                        exit_price = float(exit_row[exit_type])
                        exit_date = exit_row['date']
                    else:
                        # No valid sell OR price missing then force-close on last row of df_stock
                        last_row = df_stock.iloc[-1]
                        if not pd.isna(last_row[exit_type]):
                            exit_price = float(last_row[exit_type])
                        else:
                            # Fallback: if last row's exit_type is NaN, use its 'close'
                            exit_price = float(last_row['close'])
                        exit_date = last_row['date']

                    # Profit for a long position
                    profit = shares * (exit_price - entry_price)

                    # Record the trade
                    records.append({
                        'strategy':    strategy_name,
                        'stock':       stock,
                        'entry_date':  row_i['date'].strftime('%Y-%m-%d'),
                        'exit_date':   exit_date.strftime('%Y-%m-%d'),
                        'entry_type':  entry_type,
                        'entry_price': entry_price,
                        'exit_type':   exit_type,
                        'exit_price':  exit_price,
                        'shares':      shares,
                        'profit':      profit
                    })

                    total_profit += profit
                    total_trades += 1

            # Append summary for this strategy
            summary_rows.append({
                'strategy':     strategy_name,
                'total_profit': float(total_profit),
                'num_trades':   int(total_trades)
            })


# Run for all models
all_model_cols = [
    'xg3', 'log3', 'xg7', 'log7','nn3', 'nn7'
]

for mc in all_model_cols:
    simulate_and_record_exit_on_sell(mc)

# Add list of trades to csv
trade_df = pd.DataFrame(records, columns=[
    'strategy', 'stock', 'entry_date', 'exit_date',
    'entry_type', 'entry_price',
    'exit_type', 'exit_price',
    'shares', 'profit'
])
trade_df.to_csv(trade_list_csv, index=False)
print(f"→ Wrote {len(trade_df)} trades to '{trade_list_csv}'")

# Summary table

summary_df = pd.DataFrame(summary_rows, columns=['strategy', 'total_profit', 'num_trades'])
print(f"\nNumber of unique tickers: {num_stocks}\n")
print(summary_df.to_string(index=False))
