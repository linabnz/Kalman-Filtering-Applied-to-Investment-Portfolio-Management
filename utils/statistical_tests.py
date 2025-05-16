import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
from typing import Dict, Tuple, List, Union


def adf_test(
    series: Union[pd.Series, np.ndarray], significance_level: float = 0.05
) -> Dict:
    """
    Perform Augmented Dickey-Fuller test for stationarity.

    Args:
        series: Time series to test
        significance_level: Significance level for the test

    Returns:
        Dictionary with test results
    """
    result = adfuller(series)

    adf_output = {
        "Test Statistic": result[0],
        "p-value": result[1],
        "Critical Values": result[4],
        "Is Stationary": result[1] < significance_level,
    }

    return adf_output


def engle_granger_test(
    x: Union[pd.Series, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    significance_level: float = 0.05,
) -> Dict:
    """
    Perform Engle-Granger two-step method for testing cointegration.

    Args:
        x: First time series
        y: Second time series
        significance_level: Significance level for the test

    Returns:
        Dictionary with test results
    """

    X = np.vstack([x, np.ones(len(x))]).T
    beta = np.linalg.lstsq(X, y, rcond=None)[0]

    residuals = y - (beta[0] * x + beta[1])
    adf_result = adf_test(residuals, significance_level)

    result = {
        "Coefficient": beta[0],
        "Intercept": beta[1],
        "Residuals": residuals,
        "ADF Test": adf_result,
        "Is Cointegrated": adf_result["Is Stationary"],
    }

    return result


def find_cointegrated_pairs(
    data: pd.DataFrame, significance_level: float = 0.05
) -> List[Tuple[str, str, float]]:
    """
    Find all cointegrated pairs in a DataFrame of price series.

    Args:
        data: DataFrame where each column is a price series
        significance_level: Significance level for cointegration test

    Returns:
        List of tuples (ticker1, ticker2, p_value) for cointegrated pairs
    """
    n = data.shape[1]
    pvalue_matrix = np.ones((n, n))
    keys = data.columns
    pairs = []

    # Fill the p-value matrix
    for i in range(n):
        for j in range(i + 1, n):
            # Skip if data contains NaN values
            if np.isnan(data.iloc[:, i]).any() or np.isnan(data.iloc[:, j]).any():
                continue

            # Skip if either series is constant
            if len(data.iloc[:, i].unique()) <= 1 or len(data.iloc[:, j].unique()) <= 1:
                continue

            try:
                # Check if either series is almost constant (very low standard deviation)
                std_i = np.std(data.iloc[:, i])
                std_j = np.std(data.iloc[:, j])
                if std_i < 1e-7 or std_j < 1e-7:
                    continue

                # Perform cointegration test
                result = coint(data.iloc[:, i], data.iloc[:, j])
                pvalue_matrix[i, j] = result[1]

                # If p-value is less than significance level, add pair to result
                if result[1] < significance_level:
                    pairs.append((keys[i], keys[j], result[1]))
            except ValueError as e:
                # Skip pairs that cause errors
                print(f"Skipping pair ({keys[i]}, {keys[j]}): {str(e)}")
                continue

    return pairs
