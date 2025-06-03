import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load csv

trade_file = 'portfolio_details/trade_list.csv'
price_file = 'selected_data_with_nn.csv'

if not os.path.isfile(trade_file):
    raise SystemExit(f"Error: '{trade_file}' not found. Place it alongside this script.")
if not os.path.isfile(price_file):
    raise SystemExit(f"Error: '{price_file}' not found. Place it alongside this script.")

df_trades = pd.read_csv(trade_file, parse_dates=['entry_date', 'exit_date'])

df_prices = pd.read_csv(price_file, parse_dates=['date'])

# Strategy filter

xg7_strategies = [
    'nn3_entry_open_exit_open',
    'nn3_entry_open_exit_close',
    'nn3_entry_close_exit_open',
    'nn3_entry_close_exit_close',
    'xg7_entry_open_exit_open',
    'xg7_entry_open_exit_close',
    'xg7_entry_close_exit_open',
    'xg7_entry_close_exit_close'
]

df_trades_xg7 = df_trades[df_trades['strategy'].isin(xg7_strategies)].copy()


base_dir = 'portfolio_details'
graphs_dir = os.path.join(base_dir, 'graphs')
os.makedirs(graphs_dir, exist_ok=True)

for strategy in xg7_strategies:
    strat_dir = os.path.join(graphs_dir, strategy)
    os.makedirs(strat_dir, exist_ok=True)

    stocks = (
        df_trades_xg7[df_trades_xg7['strategy'] == strategy]['stock']
        .unique()
    )

    trades_for_strategy = df_trades_xg7[df_trades_xg7['strategy'] == strategy].copy()

    for stock in stocks:
        df_stock_prices = (
            df_prices[df_prices['stock'] == stock]
            .sort_values('date')
            .reset_index(drop=True)
        )
        if df_stock_prices.empty:
            continue 

        df_trades_stock = trades_for_strategy[trades_for_strategy['stock'] == stock].copy()

        dates = df_stock_prices['date']
        cum_profit_series = []
        for current_date in dates:
            realized = trades_for_strategy[trades_for_strategy['exit_date'] <= current_date]['profit'].sum()
            cum_profit_series.append(realized)

        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.plot(
            dates,
            df_stock_prices['close'],
            label=f'{stock} Close Price',
            color='blue'
        )
        ax1.set_xlabel('Date')
        ax1.set_ylabel(f'{stock} Price', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')


        if not df_trades_stock.empty:
            entry_dates = df_trades_stock['entry_date']
            entry_prices = df_trades_stock['entry_price']
            exit_dates = df_trades_stock['exit_date']
            exit_prices = df_trades_stock['exit_price']

            ax1.scatter(
                entry_dates,
                entry_prices,
                marker='^',
                color='green',
                label='Buy',
                zorder=5
            )
            ax1.scatter(
                exit_dates,
                exit_prices,
                marker='v',
                color='red',
                label='Exit',
                zorder=5
            )

        ax2 = ax1.twinx()
        ax2.plot(
            dates,
            cum_profit_series,
            label='Cumulative Realized Profit',
            color='orange'
        )
        ax2.set_ylabel('Cumulative Profit ($)', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        fig.suptitle(f'{strategy} · {stock}', fontsize=14, y=0.96)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        stock_chart_path = os.path.join(strat_dir, f'{stock}.png')
        plt.savefig(stock_chart_path)
        plt.close(fig)

    all_dates = pd.to_datetime(df_prices['date'].unique())
    all_dates = np.sort(all_dates)

    cum_profit_total = []
    for d in all_dates:
        total_realized = (
            trades_for_strategy[trades_for_strategy['exit_date'] <= d]['profit'].sum()
        )
        cum_profit_total.append(total_realized)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        all_dates,
        cum_profit_total,
        label='Total Cumulative Profit',
        color='purple'
    )
    ax.set_xlabel('Date')
    ax.set_ylabel('Total Cumulative Profit ($)', color='purple')
    ax.tick_params(axis='y', labelcolor='purple')
    ax.set_title(f'{strategy} · Total Portfolio', fontsize=14)
    ax.legend(loc='upper left')
    plt.tight_layout()

    total_chart_path = os.path.join(strat_dir, 'total_portfolio.png')
    plt.savefig(total_chart_path)
    plt.close(fig)
