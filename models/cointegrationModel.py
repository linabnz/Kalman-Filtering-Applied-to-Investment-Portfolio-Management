import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple
from statsmodels.tsa.stattools import adfuller
from models.kalman_filter import KalmanFilter


class BaseCointegrationModel:
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        self.is_cointegrated = False
        self.beta = None
        self.intercept = None
        self.residuals = None
        self.mu = 0
        self.sigma = 1

    def _prepare_data(
        self, y: Union[pd.Series, np.ndarray], x: Union[pd.Series, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(y, pd.Series):
            y = y.values
        if isinstance(x, pd.Series):
            x = x.values
        return y, x

    def _linear_regression(self, y: np.ndarray, x: np.ndarray) -> None:
        X = np.vstack([x, np.ones(len(x))]).T
        self.beta, self.intercept = np.linalg.lstsq(X, y, rcond=None)[0]
        self.residuals = y - (self.beta * x + self.intercept)
        self.mu = np.mean(self.residuals)
        self.sigma = np.std(self.residuals)
        if self.sigma == 0:
            self.is_cointegrated = False
        else:
            adf_result = adfuller(self.residuals)
            self.is_cointegrated = adf_result[1] < self.significance_level


class CointegrationModel(BaseCointegrationModel):
    def fit(
        self, y: Union[pd.Series, np.ndarray], x: Union[pd.Series, np.ndarray]
    ) -> "CointegrationModel":
        y, x = self._prepare_data(y, x)
        self._linear_regression(y, x)
        return self


class PartialCointegrationModel(BaseCointegrationModel):
    def __init__(self, significance_level: float = 0.05):
        super().__init__(significance_level)
        self.rho = None
        self.sigma_mr = None
        self.sigma_rw = None
        self.kalman_gain = None
        self.mean_reverting_component = None
        self.random_walk_component = None
        self.kalman_filter = None

    def fit(
        self, y: Union[pd.Series, np.ndarray], x: Union[pd.Series, np.ndarray]
    ) -> "PartialCointegrationModel":
        y, x = self._prepare_data(y, x)
        self._linear_regression(y, x)

        if self.is_cointegrated:
            self.rho, self.sigma_mr, self.sigma_rw, self.kalman_gain = (
                KalmanFilter.estimate_parameters(self.residuals)
            )
            self.kalman_filter = KalmanFilter(
                rho=self.rho, kalman_gain=self.kalman_gain
            )
            self.mean_reverting_component, self.random_walk_component = (
                self.kalman_filter.filter(self.residuals)
            )
        return self

    def update_components(self, new_spread: float) -> Tuple[float, float]:
        if not all(
            [
                self.kalman_filter,
                self.mean_reverting_component,
                self.random_walk_component,
            ]
        ):
            raise ValueError("Model not fitted or Kalman filter not applied.")

        last_mr = self.mean_reverting_component[-1]
        last_rw = self.random_walk_component[-1]
        mr_pred = self.rho * last_mr
        rw_pred = last_rw
        innovation = new_spread - (mr_pred + rw_pred)
        new_mr = mr_pred + self.kalman_gain * innovation
        new_rw = rw_pred + (1 - self.kalman_gain) * innovation
        return new_mr, new_rw

    def get_mean_reverting_proportion(self) -> float:
        if None in (self.sigma_mr, self.sigma_rw, self.rho):
            raise ValueError("Kalman parameters not estimated.")
        numerator = 2 * self.sigma_mr**2
        denominator = 2 * self.sigma_mr**2 + (1 + self.rho) * self.sigma_rw**2
        return numerator / denominator
