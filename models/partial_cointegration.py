import numpy as np
import pandas as pd
from typing import Tuple, Dict, Union, Optional
from statsmodels.tsa.stattools import adfuller
from models.kalman_filter import KalmanFilter


class PartialCointegrationModel:
    """
    Model for partial cointegration-based statistical arbitrage using Kalman filtering to separate
    the mean-reverting and random walk components of the spread.
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize PartialCointegrationModel.
        
        Args:
            significance_level: Significance level for cointegration tests
        """
        self.significance_level = significance_level
        self.beta = None
        self.intercept = None
        self.residuals = None
        self.is_cointegrated = False
        self.z_scores = None
        self.mu = 0
        self.sigma = 1
        
        # Partial cointegration specific parameters
        self.rho = None  # Mean reversion coefficient
        self.sigma_mr = None  # Std dev of mean-reverting innovations
        self.sigma_rw = None  # Std dev of random walk innovations
        self.kalman_gain = None  # Kalman filter gain
        self.mean_reverting_component = None  # Mean-reverting component of spread
        self.random_walk_component = None  # Random walk component of spread
        self.kalman_filter = None  # Kalman filter instance
    
    def fit(self, y: Union[pd.Series, np.ndarray], x: Union[pd.Series, np.ndarray]) -> 'PartialCointegrationModel':
        """
        Fit cointegration model where y = beta * x + intercept + residuals.
        Then decomposes residuals into mean-reverting and random walk components.
        
        Args:
            y: Dependent variable (price series)
            x: Independent variable (price series)
            
        Returns:
            Self
        """
        
        if isinstance(y, pd.Series):
            y = y.values
        if isinstance(x, pd.Series):
            x = x.values
        
        # Add constant term for intercept
        X = np.vstack([x, np.ones(len(x))]).T
        
        # Estimate beta and intercept
        beta, intercept = np.linalg.lstsq(X, y, rcond=None)[0]
        
        self.beta = beta
        self.intercept = intercept
        self.residuals = y - (beta * x + intercept)
        
        # Calculate basic statistics for total residuals
        self.mu = np.mean(self.residuals)
        self.sigma = np.std(self.residuals)
        
        # Test for cointegration
        adf_result = adfuller(self.residuals)
        self.is_cointegrated = adf_result[1] < self.significance_level
        
        # If cointegrated, estimate partial cointegration parameters and apply Kalman filter
        if self.is_cointegrated:
            # Estimate Kalman filter parameters from residuals
            self.rho, self.sigma_mr, self.sigma_rw, self.kalman_gain = KalmanFilter.estimate_parameters(self.residuals)
            
            # Create and apply Kalman filter
            self.kalman_filter = KalmanFilter(rho=self.rho, kalman_gain=self.kalman_gain)
            self.mean_reverting_component, self.random_walk_component = self.kalman_filter.filter(self.residuals)
        
        return self
    
    def calculate_z_scores(self, window: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate z-scores for the mean-reverting component using rolling mean and standard deviation.
        
        Args:
            window: Window size for rolling statistics
            
        Returns:
            Tuple of (mean-reverting component z-scores, total residual z-scores)
        """
        if self.mean_reverting_component is None:
            raise ValueError("Model not fitted yet or Kalman filter not applied")
        
        # Convert to pandas Series for rolling calculations
        mr_series = pd.Series(self.mean_reverting_component)
        residuals_series = pd.Series(self.residuals)
        
        # Calculate rolling mean and standard deviation for mean-reverting component
        mr_rolling_mean = mr_series.rolling(window=window).mean()
        mr_rolling_std = mr_series.rolling(window=window).std()
        
        # Calculate z-scores for mean-reverting component
        mr_z_scores = (mr_series - mr_rolling_mean) / mr_rolling_std
        
        # Calculate z-scores for total residuals (for comparison)
        residuals_rolling_mean = residuals_series.rolling(window=window).mean()
        residuals_rolling_std = residuals_series.rolling(window=window).std()
        residuals_z_scores = (residuals_series - residuals_rolling_mean) / residuals_rolling_std
        
        # Fill NaN values at the beginning
        mr_z_scores = mr_z_scores.fillna(0).values
        residuals_z_scores = residuals_z_scores.fillna(0).values
        
        return mr_z_scores, residuals_z_scores
    
    def predict(self, x: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """
        Predict y values based on the fitted model.
        
        Args:
            x: Independent variable values
            
        Returns:
            Predicted y values
        """
        if self.beta is None or self.intercept is None:
            raise ValueError("Model not fitted yet")
        
        return self.beta * x + self.intercept
    
    def predict_spread(self, y: Union[pd.Series, np.ndarray], x: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """
        Calculate the spread between actual y values and predicted y values.
        
        Args:
            y: Actual dependent variable values
            x: Independent variable values
            
        Returns:
            Spread values (residuals)
        """
        predicted_y = self.predict(x)
        return y - predicted_y
    
    def update_components(self, new_spread: float) -> Tuple[float, float]:
        """
        Update mean-reverting and random walk components for a new spread value.
        
        Args:
            new_spread: New spread value
            
        Returns:
            Tuple of (new_mean_reverting_component, new_random_walk_component)
        """
        if self.kalman_filter is None or self.mean_reverting_component is None or self.random_walk_component is None:
            raise ValueError("Model not fitted yet or Kalman filter not applied")
        
        # Get last values of components
        last_mr = self.mean_reverting_component[-1]
        last_rw = self.random_walk_component[-1]
        
        # Prediction step
        mr_pred = self.rho * last_mr
        rw_pred = last_rw
        
        # Calculate innovation
        innovation = new_spread - (mr_pred + rw_pred)
        
        # Update step
        new_mr = mr_pred + self.kalman_gain * innovation
        new_rw = rw_pred + (1 - self.kalman_gain) * innovation
        
        return new_mr, new_rw
    
    def get_mean_reverting_proportion(self) -> float:
        """
        Calculate the proportion of variance attributable to the mean-reverting component.
        Based on equation 19 from the paper.
        
        Returns:
            Proportion of variance attributable to mean reversion (R²_mr)
        """
        if self.sigma_mr is None or self.sigma_rw is None or self.rho is None:
            raise ValueError("Partial cointegration parameters not estimated yet")
        
        numerator = 2 * self.sigma_mr**2
        denominator = 2 * self.sigma_mr**2 + (1 + self.rho) * self.sigma_rw**2
        
        return numerator / denominator