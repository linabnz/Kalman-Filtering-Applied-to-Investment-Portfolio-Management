import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union


def load_stock_data(file_path: str) -> pd.DataFrame:
    """
    Load stock price data from a CSV file.
    
    Args:
        file_path: Path to the CSV file containing stock price data
        
    Returns:
        DataFrame with stock price data
    """
    try:
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        print(f"Successfully loaded data with shape {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise


def prepare_price_series(data: pd.DataFrame, tickers: List[str], start_date=None, end_date=None) -> Dict[str, pd.Series]:
    """
    Prepare price series for a list of tickers within specified date range.
    
    Args:
        data: DataFrame with stock price data
        tickers: List of stock tickers to extract
        start_date: Start date for the data (optional)
        end_date: End date for the data (optional)
        
    Returns:
        Dictionary mapping each ticker to its price series
    """
    result = {}
    
    # Filter by date range if provided
    if start_date is not None or end_date is not None:
        mask = True
        if start_date is not None:
            mask = mask & (data.index >= start_date)
        if end_date is not None:
            mask = mask & (data.index <= end_date)
        data = data.loc[mask]
    
    # Extract price series for each ticker
    for ticker in tickers:
        if ticker in data.columns:
            result[ticker] = data[ticker].copy()
        else:
            print(f"Warning: Ticker {ticker} not found in data")
    
    return result


def split_data(data: pd.DataFrame, train_ratio: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and testing sets based on time.
    
    Args:
        data: DataFrame with time series data
        train_ratio: Proportion of data to use for training
        
    Returns:
        Tuple of (training_data, testing_data)
    """
    split_idx = int(len(data) * train_ratio)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    
    return train_data, test_data


def calculate_returns(prices: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
    """
    Calculate percentage returns from price series.
    
    Args:
        prices: Price series or DataFrame of price series
        
    Returns:
        Returns series or DataFrame
    """
    return prices.pct_change().fillna(0)


def normalize_prices(prices: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
    """
    Normalize prices to start at 1.
    
    Args:
        prices: Price series or DataFrame of price series
        
    Returns:
        Normalized prices
    """
    return prices / prices.iloc[0]