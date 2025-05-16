import numpy as np
import pandas as pd
from typing import Tuple, Dict, Union, Optional
from statsmodels.tsa.stattools import adfuller


import numpy as np
import pandas as pd
from typing import Union, Optional
from statsmodels.tsa.stattools import adfuller


class CointegrationModel:
    """
    Dynamically updatable model for cointegration-based statistical arbitrage.
    """

    def __init__(self, significance_level: float = 0.05, z_window: int = 20):
        self.significance_level = significance_level
        self.beta = None
        self.intercept = None
        self.is_cointegrated = False
        self.mu = None
        self.sigma = None

    def fit(
        self, y: Union[pd.Series, np.ndarray], x: Union[pd.Series, np.ndarray]
    ) -> "CointegrationModel":
        """
        Fit cointegration model where y = beta * x + intercept + residuals.
        """

        if isinstance(y, pd.Series):
            y = y.values
        if isinstance(x, pd.Series):
            x = x.values

        X = np.vstack([x, np.ones(len(x))]).T
        beta, intercept = np.linalg.lstsq(X, y, rcond=None)[0]

        self.beta = beta
        self.intercept = intercept
        residuals = y - (beta * x + intercept)
        self.residuals = list(residuals)
        self.mu = np.mean(residuals)
        self.sigma = np.std(residuals)

        # Stationarity test (ADF)
        if np.std(residuals) == 0:
            self.is_cointegrated = False
            return self
        adf_result = adfuller(residuals)
        self.is_cointegrated = adf_result[1] < self.significance_level

        return self

    def predict(self, x: Union[pd.Series, np.ndarray]) -> np.ndarray:
        if self.beta is None or self.intercept is None:
            raise ValueError("Model not fitted yet")
        return self.beta * x + self.intercept

    def predict_spread(
        self, y: Union[pd.Series, np.ndarray], x: Union[pd.Series, np.ndarray]
    ) -> np.ndarray:
        predicted_y = self.predict(x)
        return y - predicted_y
