# This file plots individial tickers 
# Change code at bottom to olot using ticker name and dates

import pandas as pd
import matplotlib.pyplot as plt

def plot_candlestick_with_indicators(csv_file, symbol, start_date, end_date):
    # Load dates
    df = pd.read_csv(csv_file, parse_dates=['date'])
    
    # Sort using ticker & date
    mask = (
        (df['date'] >= pd.to_datetime(start_date)) &
        (df['date'] <= pd.to_datetime(end_date)) &
        (df['Name'] == symbol)
    )
    df = df.loc[mask].copy()
    
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['x'] = df.index

    # plot price, volume histogram, MACD
    fig, (ax_price, ax_vol, ax_macd) = plt.subplots(
        3, 1,
        sharex=True,
        figsize=(12, 9),
        gridspec_kw={'height_ratios': [3, 1, 1], 'hspace': 0.05}
    )

    # Create candlesticks
    width = 0.6
    for _, row in df.iterrows():
        color = 'green' if row.close >= row.open else 'red'
        # wick
        ax_price.vlines(row.x, row.low, row.high, color=color, linewidth=1)
        # body
        bottom = min(row.open, row.close)
        height = abs(row.close - row.open)
        ax_price.bar(row.x, height, bottom=bottom,
                     width=width, color=color, align='center')

    # Plot moving averages
    ax_price.plot(df.x, df['sma_7'],  label='SMA (7)',  linewidth=1.2)
    ax_price.plot(df.x, df['ema_14'], label='EMA (14)', linewidth=1.2)
    ax_price.set_ylabel('Price')
    ax_price.legend(loc='upper left', fontsize='small')
    ax_price.grid(True, linestyle=':', linewidth=0.5)

    # Volume histogram
    ax_vol.bar(df.x, df['volume'], width=width, align='center')
    ax_vol.set_ylabel('Volume')
    ax_vol.grid(True, linestyle=':', linewidth=0.5)

    # MACD & signal line
    ax_macd.plot(df.x, df['macd'],        label='MACD (12–26)', linewidth=1.2)
    ax_macd.plot(df.x, df['macd_signal'], label='Signal (9)',   linewidth=1.2)
    ax_macd.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax_macd.set_ylabel('MACD')
    ax_macd.legend(loc='upper left', fontsize='small')
    ax_macd.grid(True, linestyle=':', linewidth=0.5)

    # Relabel x-axis with dates
    n = len(df)
    step = max(1, n // 10)
    ticks  = df.x[::step]
    labels = df.date.dt.strftime('%Y-%m-%d')[::step]
    ax_macd.set_xticks(ticks)
    ax_macd.set_xticklabels(labels, rotation=30, ha='right')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # ─── User parameters ───────────────────────
    CSV_FILE   = 'stocks_with_indicators.csv'
    SYMBOL     = 'MSFT'                   
    START_DATE = '2013-02-08'
    END_DATE   = '2018-02-07'
    # ───────────────────────────────────────────

    plot_candlestick_with_indicators(
        CSV_FILE, SYMBOL, START_DATE, END_DATE
    )
