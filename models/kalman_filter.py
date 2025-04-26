# models/kalman_filter.py
import numpy as np

class KalmanFilter:
    def __init__(self, rho=0.5, sigma_mr=1.0, sigma_rw=1.0):
        self.rho = rho  # Mean reversion coefficient
        self.sigma_mr = sigma_mr  # Standard deviation of mean-reverting noise
        self.sigma_rw = sigma_rw  # Standard deviation of random walk noise
        self.kalman_gain = None
        self.hidden_states = None
        
    def calculate_kalman_gain(self):
        """Calculate the Kalman gain based on model parameters."""
        rho = self.rho
        sigma_mr = self.sigma_mr
        sigma_rw = self.sigma_rw
        
        # Calculate Kalman gain using the formula from the paper
        numerator = 2 * sigma_mr**2
        term1 = sigma_rw * np.sqrt((rho+1)**2 * sigma_rw**2 + 4 * sigma_mr**2)
        term2 = rho * sigma_rw + sigma_rw + 2 * sigma_mr**2
        denominator = term1 + term2
        
        self.kalman_gain = numerator / denominator
        
        return self.kalman_gain
    
    def filter(self, spread):
        """Apply Kalman filter to decompose spread into mean-reverting and random walk components."""
        n = len(spread)
        mean_reverting = np.zeros(n)
        random_walk = np.zeros(n)
        
        # Initialize with first observation
        random_walk[0] = spread[0]
        
        # Calculate Kalman gain if not already calculated
        if self.kalman_gain is None:
            self.calculate_kalman_gain()
        
        # Apply Kalman filter
        for t in range(1, n):
            # Prediction step
            mean_reverting_pred = self.rho * mean_reverting[t-1]
            random_walk_pred = random_walk[t-1]
            spread_pred = mean_reverting_pred + random_walk_pred
            
            # Update step using Kalman gain
            innovation = spread[t] - spread_pred
            mean_reverting[t] = mean_reverting_pred + self.kalman_gain * innovation
            random_walk[t] = random_walk_pred + (1 - self.kalman_gain) * innovation
        
        self.hidden_states = {'mean_reverting': mean_reverting, 'random_walk': random_walk}
        return mean_reverting, random_walk