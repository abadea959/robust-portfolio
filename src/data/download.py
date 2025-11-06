import os
import pandas as pd
import yfinance as yf


def download_adj_close(tickers, start, end=None):
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
        group_by="column",
    )

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Adj Close"].copy()
    else:
        prices = data[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})

    prices = prices.dropna(how="all")
    prices = prices.ffill()
    prices = prices.dropna(how="any")

    prices.index = pd.to_datetime(prices.index).tz_localize(None)

    return prices


def cache_prices_csv(df, path="data/prices.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=True)


if __name__ == "__main__":
    from src.config import load_config

    cfg = load_config()
    tickers = cfg["tickers"]
    start = cfg["start"]
    end = cfg["end"]

    print(f"Downloading {tickers} from {start} to {end or 'today'} ...")
    prices = download_adj_close(tickers, start, end)
    print(f"Downloaded shape: {prices.shape}")
    print(prices.tail(3))
    cache_prices_csv(prices)
    print("Saved to data/prices.csv")
