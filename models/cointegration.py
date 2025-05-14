import numpy as np
import pandas as pd
from typing import Tuple, Dict, Union, Optional
from statsmodels.tsa.stattools import adfuller


class CointegrationModel:
    """
    Model for cointegration-based statistical arbitrage.
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize CointegrationModel.
        
        Args:
            significance_level: Significance level for cointegration tests
        """
        self.significance_level = significance_level
        self.beta = None
        self.intercept = None
        self.residuals = None
        self.is_cointegrated = False
        self.z_scores = None
    
    def fit(self, y: Union[pd.Series, np.ndarray], x: Union[pd.Series, np.ndarray]) -> 'CointegrationModel':
        """
        Fit cointegration model where y = beta * x + intercept + residuals.
        
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
        
     
        beta, intercept = np.linalg.lstsq(X, y, rcond=None)[0]
        
        self.beta = beta
        self.intercept = intercept
        self.residuals = y - (beta * x + intercept)
        
        
        adf_result = adfuller(self.residuals)
        self.is_cointegrated = adf_result[1] < self.significance_level
        
        return self
    
    def calculate_z_scores(self, window: int = 20) -> np.ndarray:
        """
        Calculate z-scores for the residuals using rolling mean and standard deviation.
        
        Args:
            window: Window size for rolling statistics
            
        Returns:
            Array of z-scores
        """
        if self.residuals is None:
            raise ValueError("Model not fitted yet")
        
        # Convert to pandas Series for rolling calculations
        residuals_series = pd.Series(self.residuals)
        
        # Calculate rolling mean and standard deviation
        rolling_mean = residuals_series.rolling(window=window).mean()
        rolling_std = residuals_series.rolling(window=window).std()
        
        # Calculate z-scores
        z_scores = (residuals_series - rolling_mean) / rolling_std
        
        # Fill NaN values at the beginning
        z_scores = z_scores.fillna(0)
        
        self.z_scores = z_scores.values
        return self.z_scores
    
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