import os
import time
import logging
from datetime import date
from typing import Tuple, List

import pandas as pd
import yfinance as yf


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_option_chain(
    ticker: yf.Ticker,
    expiry: str,
    pause: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch option chain for a single expiration date.

    Args:
        ticker: yfinance Ticker object
        expiry: expiration date string
        pause: delay between API calls

    Returns:
        Tuple of (calls_df, puts_df)
    """
    try:
        option_chain = ticker.option_chain(expiry)
        calls = option_chain.calls.copy()
        puts = option_chain.puts.copy()

        calls["expiration"] = expiry
        puts["expiration"] = expiry

        time.sleep(pause)
        return calls, puts

    except Exception as e:
        logger.warning(f"Failed to fetch expiry {expiry}: {e}")
        return pd.DataFrame(), pd.DataFrame()


def fetch_all_options(
    symbol: str,
    output_dir: str,
    pause: float = 0.5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download all Yahoo Finance option chains for a given symbol.

    Saves aggregated calls and puts into CSV files.

    Args:
        symbol: Ticker symbol (e.g., 'AAPL')
        output_dir: Directory to save output files
        pause: Delay between API calls to avoid rate limiting

    Returns:
        Tuple of (calls_df, puts_df)
    """
    ticker = yf.Ticker(symbol)
    expiries: List[str] = ticker.options

    if not expiries:
        raise ValueError(f"No options found for symbol: {symbol}")

    logger.info(f"{symbol}: {len(expiries)} expiries found")

    all_calls: List[pd.DataFrame] = []
    all_puts: List[pd.DataFrame] = []

    os.makedirs(output_dir, exist_ok=True)

    for expiry in expiries:
        calls, puts = fetch_option_chain(ticker, expiry, pause)

        if not calls.empty:
            all_calls.append(calls)

        if not puts.empty:
            all_puts.append(puts)

    if not all_calls or not all_puts:
        raise RuntimeError("No option data could be fetched.")

    calls_df = pd.concat(all_calls, ignore_index=True)
    puts_df = pd.concat(all_puts, ignore_index=True)

    save_option_data(calls_df, puts_df, symbol, output_dir)

    logger.info(f"All option data for {symbol} saved in {output_dir}")

    return calls_df, puts_df


def save_option_data(
    calls_df: pd.DataFrame,
    puts_df: pd.DataFrame,
    symbol: str,
    output_dir: str
) -> None:
    """
    Save calls and puts DataFrames to CSV files.

    Args:
        calls_df: Calls DataFrame
        puts_df: Puts DataFrame
        symbol: Ticker symbol
        output_dir: Directory to save files
    """
    today_str = date.today().isoformat()

    calls_path = os.path.join(output_dir, f"{symbol}_calls_{today_str}.csv")
    puts_path = os.path.join(output_dir, f"{symbol}_puts_{today_str}.csv")

    calls_df.to_csv(calls_path, index=False)
    puts_df.to_csv(puts_path, index=False)

    logger.info(f"Saved calls to {calls_path}")
    logger.info(f"Saved puts to {puts_path}")