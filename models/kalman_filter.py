import numpy as np
import pandas as pd

class KalmanFilter:
    
    
    def __init__(self, rho=0.1, sigma2_mr=1.0, sigma2_rw=0.5):
        
        self.rho = rho
        self.sigma2_mr = sigma2_mr
        self.sigma2_rw = sigma2_rw
        self.K_mr, self.K_rw = self._calculate_kalman_gains()
    
    def _calculate_kalman_gains(self):
        
        num = 2 * self.sigma2_mr
        sqrt_term = np.sqrt(((self.rho + 1)**2) * self.sigma2_rw**2 + 4 * self.sigma2_mr**2)
        denom1 = self.sigma2_rw * sqrt_term + self.rho * self.sigma2_rw + self.sigma2_rw
        denom2 = self.sigma2_rw * sqrt_term - self.rho * self.sigma2_rw + self.sigma2_rw
        K_mr = num / denom1
        K_rw = num / denom2
        return K_mr, K_rw
    
    def decompose(self, spread):
        
        n = len(spread)
        epsilon_mr = np.zeros(n)
        epsilon_rw = np.zeros(n)
        epsilon_rw[0] = spread.iloc[0]
        
        for t in range(1, n):
            pred_mr = self.rho * epsilon_mr[t - 1]
            pred_rw = epsilon_rw[t - 1]
            error = spread.iloc[t] - (pred_mr + pred_rw)
            
            epsilon_mr[t] = pred_mr + self.K_mr * error
            epsilon_rw[t] = pred_rw + self.K_rw * error
        
        return pd.Series(epsilon_mr, index=spread.index), pd.Series(epsilon_rw, index=spread.index)