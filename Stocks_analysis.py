import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from prophet import Prophet

DATA_FILE = "C:\\Users\\jayde\\OneDrive\\Attachments\\Documents\\Desktop\\Nifty50_stock_10y.csv"
STOCKS = ['NIFTY50', 'Tata_Steel', 'Reliance',
          'Hindustan_Unilever', 'Tata_Communcations', 'Finolex_Cables']
MA_WINDOW = 20
MAX_LAGS = 60

def load_data(path):
    df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
    return df.dropna(how='all')

def apply_moving_average(df, stocks, window):
    out = pd.DataFrame(index=df.index)
    for s in stocks:
        out[s] = df[s]
        out[f'Filtered_{s}'] = df[s].rolling(window=window).mean()
    return out.dropna()

def plot_series(date_index, series1, series2, title, ylabel='Value'):
    plt.figure(figsize=(10, 4))
    plt.plot(date_index, series1, label='Original', alpha=0.8)
    plt.plot(date_index, series2, label='Filtered (MA {})'.format(MA_WINDOW), linewidth=1.5)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_acfs(series_orig, series_filt, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    plot_acf(series_orig, ax=axes[0], lags=MAX_LAGS, title='ACF - Original')
    plot_acf(series_filt, ax=axes[1], lags=MAX_LAGS, title='ACF - Filtered')
    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_ccf(x, y, title):
    fig, axes = plt.subplots(1, 1, figsize=(10, 3.5))
    axes.xcorr(x - np.nanmean(x), y - np.nanmean(y), maxlags=MAX_LAGS, usevlines=True)
    axes.set_title(title)
    axes.grid(True)
    plt.tight_layout()
    plt.show()

def prophet_one_step(ds, y, name):
    df_prophet = pd.DataFrame({'ds': ds, 'y': y}).dropna()
    if len(df_prophet) < 10:
        print(f"Not enough rows for Prophet for {name}. Skipping.")
        return
    m = Prophet(daily_seasonality=True, weekly_seasonality=False, yearly_seasonality=False)
    m.fit(df_prophet)
    future = m.make_future_dataframe(periods=1, freq='D')
    forecast = m.predict(future)
    last_two = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(2)
    print(f"\nProphet 1-step forecast for {name}:\n", last_two.to_string(index=False))
    _ = m.plot(forecast)
    plt.title(f'Prophet Forecast - {name}')
    plt.show()

def run_price_workflow(df, stocks):
    df_ma = apply_moving_average(df, stocks, MA_WINDOW)

    for s in stocks:
        orig = df_ma[s]
        filt = df_ma[f'Filtered_{s}']
        plot_series(df_ma.index, orig, filt, f'{s} - Price vs Filtered', ylabel='Adjusted Close')
        plot_acfs(orig, filt, f'ACF Comparison - {s}')
    # CCFs relative to NIFTY50
    base = 'NIFTY50'
    for s in stocks:
        if s == base:
            continue
        # align and drop NaNs before ccf
        a = df_ma[f'Filtered_{s}'].dropna()
        b = df_ma[f'Filtered_{base}'].dropna()
        common_idx = a.index.intersection(b.index)
        plot_ccf(a.loc[common_idx].values, b.loc[common_idx].values,
                 f'CCF - Filtered {s} vs Filtered {base}')
    # Prophet on each stock's adjusted close (one-step)
    for s in stocks:
        prophet_one_step(df_ma.index, df_ma[s].values, f'Price - {s}')

def run_return_workflow(df, stocks):
    df_ma = apply_moving_average(df, stocks, MA_WINDOW)

    # compute returns for original and filtered
    df_ret = pd.DataFrame(index=df_ma.index)
    for s in stocks:
        df_ret[f'Return_{s}'] = df_ma[s].pct_change()
        df_ret[f'Filtered_Return_{s}'] = df_ma[f'Filtered_{s}'].pct_change()
    df_ret = df_ret.dropna()

    for s in stocks:
        orig = df_ret[f'Return_{s}']
        filt = df_ret[f'Filtered_Return_{s}']
        plot_series(df_ret.index, orig, filt, f'{s} - Return vs Filtered Return', ylabel='Daily Return')
        plot_acfs(orig, filt, f'ACF Comparison (Returns) - {s}')
    # CCFs on returns relative to NIFTY50 returns
    base = 'NIFTY50'
    for s in stocks:
        if s == base:
            continue
        a = df_ret[f'Filtered_Return_{s}'].dropna()
        b = df_ret[f'Filtered_Return_{base}'].dropna()
        common_idx = a.index.intersection(b.index)
        plot_ccf(a.loc[common_idx].values, b.loc[common_idx].values,
                 f'CCF (Returns) - Filtered_Return_{s} vs Filtered_Return_{base}')
    # Prophet on returns
    for s in stocks:
        prophet_one_step(df_ret.index, df_ret[f'Return_{s}'].values, f'Return - {s}')

def main():
    df = load_data(DATA_FILE)
    # ensure required columns exist
    missing = [c for c in STOCKS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    run_price_workflow(df, STOCKS)
    run_return_workflow(df, STOCKS)

if __name__ == "__main__":
    main()
