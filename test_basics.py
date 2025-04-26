import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.data_loader import load_stock_data, split_data
from utils.statistical_tests import find_cointegrated_pairs
from models.cointegration import CointegrationModel

def generate_test_data():
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=500, freq='D')
    
    
    random_walk = np.random.standard_normal(size=500).cumsum()

    stock1 = random_walk + np.random.normal(0, 1, 500)
    stock2 = 0.7 * random_walk + np.random.normal(0, 0.5, 500)
    
    
    stock3 = np.random.standard_normal(size=500).cumsum() + np.random.normal(0, 1, 500)
    
    df = pd.DataFrame({
        'STOCK1': stock1,
        'STOCK2': stock2,
        'STOCK3': stock3
    }, index=dates)
    
    return df

def main():
    print("Testing basic functionality...")
    
    data = generate_test_data()
    print(f"Generated test data with shape {data.shape}")
    
  
    train_data, test_data = split_data(data, train_ratio=0.7)
    print(f"Split data into training set ({train_data.shape}) and testing set ({test_data.shape})")
    
    cointegrated_pairs = find_cointegrated_pairs(train_data)
    print(f"Found {len(cointegrated_pairs)} cointegrated pairs:")
    for pair in cointegrated_pairs:
        print(f"  {pair[0]} and {pair[1]} (p-value: {pair[2]:.6f})")
    
    
    if len(cointegrated_pairs) > 0:
        ticker1, ticker2, _ = cointegrated_pairs[0]
        model = CointegrationModel()
        model.fit(train_data[ticker1], train_data[ticker2])
        print(f"Fitted cointegration model: y = {model.beta:.4f} * x + {model.intercept:.4f}")
        print(f"Is cointegrated: {model.is_cointegrated}")
        
        z_scores = model.calculate_z_scores(window=20)
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(3, 1, 1)
        plt.plot(train_data.index, train_data[ticker1], label=ticker1)
        plt.plot(train_data.index, train_data[ticker2], label=ticker2)
        plt.title('Price Series')
        plt.legend()
        
        plt.subplot(3, 1, 2)
        plt.plot(train_data.index, model.residuals)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.title('Residuals (Spread)')
        
        plt.subplot(3, 1, 3)
        plt.plot(train_data.index, z_scores)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.axhline(y=2, color='g', linestyle='--')
        plt.axhline(y=-2, color='g', linestyle='--')
        plt.title('Z-scores')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()