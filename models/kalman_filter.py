# models/kalman_filter.py
import numpy as np

class KalmanFilter:
    """
    Implementation of a Kalman filter for decomposing a signal into mean-reverting and random walk components.
    """
    
    def __init__(self, rho: float = 0.5, kalman_gain: float = 0.5):
        """
        Initialize KalmanFilter.
        
        Args:
            rho: Mean reversion coefficient (-1 < rho < 1)
            kalman_gain: Kalman filter gain (0 <= kalman_gain <= 1)
        """
        self.rho = max(min(rho, 0.999), -0.999)  
        self.kalman_gain = max(min(kalman_gain, 1.0), 0.0)  
        
    def filter(self, signal: np.ndarray) -> tuple:
        """
        Apply Kalman filter to decompose a signal into mean-reverting and random walk components.
        
        Args:
            signal: Input signal to decompose
            
        Returns:
            Tuple of (mean_reverting_component, random_walk_component)
        """
        n = len(signal)
        
      
        mean_reverting = np.zeros(n)
        random_walk = np.zeros(n)
        
       
        random_walk[0] = signal[0]
        
      
        for t in range(1, n):
            # Prediction step
            mean_reverting_pred = self.rho * mean_reverting[t-1]
            random_walk_pred = random_walk[t-1]
            signal_pred = mean_reverting_pred + random_walk_pred
            
            
            innovation = signal[t] - signal_pred
            
            # Update step
            mean_reverting[t] = mean_reverting_pred + self.kalman_gain * innovation
            random_walk[t] = random_walk_pred + (1 - self.kalman_gain) * innovation
        
        return mean_reverting, random_walk
    
    @staticmethod
    def estimate_parameters(signal: np.ndarray) -> tuple:
        """
        Estimate optimal parameters (rho, sigma_mr, sigma_rw) for Kalman filtering
        based on signal autocovariances.
        
        Args:
            signal: Input signal
            
        Returns:
            Tuple of (rho, sigma_mr, sigma_rw, kalman_gain)
        """
     
        signal = np.asarray(signal)
        n = len(signal)
        
      
        signal_mean = np.mean(signal)
        signal_centered = signal - signal_mean
        
        
        v1 = np.dot(signal_centered[1:], signal_centered[:-1]) / (n - 1)
        v2 = np.dot(signal_centered[2:], signal_centered[:-2]) / (n - 2)
        v3 = np.dot(signal_centered[3:], signal_centered[:-3]) / (n - 3)
        
        
        denominator = 2*v1 - v2
        
        
        if abs(denominator) < 1e-10:
            rho = 0.5  
        else:
            rho = -(v1 - 2*v2 + v3) / denominator
        
      
        rho = max(min(rho, 0.999), -0.999)
        
        
        term1 = (rho + 1) / (rho - 1)
        term2 = v2 - 2*v1
        
        
        if term1 * term2 <= 0:
            sigma_mr = np.sqrt(0.1 * np.var(signal))  # Fallback value
        else:
            sigma_mr = np.sqrt(0.5 * term1 * term2)
            
        
        sigma_rw_squared = 0.5 * (v2 - 2 * (sigma_mr**2))
        
       
        if sigma_rw_squared <= 0:
            sigma_rw_squared = 0.1 * np.var(signal)  # Fallback value
            
        sigma_rw = np.sqrt(sigma_rw_squared)
        
        
        term_under_sqrt = (rho + 1) * 2 * sigma_rw**2 + 4 * sigma_mr**2
        
       
        if term_under_sqrt < 0:
            term_under_sqrt = 4 * sigma_mr**2  # Fallback value
            
        numerator = 2 * sigma_mr**2
        denominator = np.sqrt(term_under_sqrt) + rho * sigma_rw + sigma_rw + 2 * sigma_mr**2
        
       
        if denominator < 1e-10:
            kalman_gain = 0.5  # Default value
        else:
            kalman_gain = numerator / denominator
            
       
        kalman_gain = max(min(kalman_gain, 1.0), 0.0)
        
        return rho, sigma_mr, sigma_rw, kalman_gain