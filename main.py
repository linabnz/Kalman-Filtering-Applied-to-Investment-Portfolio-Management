# main.py
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from models.cointegration import CointegrationModel
from models.partial_cointegration import PartialCointegrationModel
from models.kalman_filter import KalmanFilter
from utils.data_loader import load_stock_data
from strategies.pairs_trading import PairsTrading
from strategies.statistical_arbitrage import StatisticalArbitrage

def main():
    parser = argparse.ArgumentParser(description='Kalman Filter Portfolio Management')
    parser.add_argument('--data_path', type=str, default='data/stocks.csv', help='Path to stock price data')
    parser.add_argument('--strategy', type=str, default='partial_coint', 
                        choices=['pairs', 'coint', 'partial_coint', 'cnn_rl'],
                        help='Trading strategy to use')
    parser.add_argument('--viz', action='store_true', help='Visualize results')
    args = parser.parse_args()
    
  
    data = load_stock_data(args.data_path)
    
   
    if args.strategy == 'pairs':
        strategy = PairsTrading(significance_level=0.05, z_threshold=2.0)
    elif args.strategy == 'coint':
        strategy = StatisticalArbitrage(model='cointegration', z_threshold=2.0)
    elif args.strategy == 'partial_coint':
        strategy = StatisticalArbitrage(model='partial_cointegration', 
                                        rho=0.7, kalman_gain=0.7, z_threshold=1.25)
    elif args.strategy == 'cnn_rl':
        
        pass
    

    portfolio_value, trades = strategy.backtest(data)
    

    print(f"Strategy: {args.strategy}")
    print(f"Final portfolio value: {portfolio_value[-1]:.2f}")
    print(f"Number of trades: {len(trades)}")
    print(f"Sharpe ratio: {strategy.calculate_sharpe_ratio():.2f}")
    
   
    if args.viz:
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_value, label=f'{args.strategy} strategy')
        plt.plot(data['market_index'], label='Market Index')
        plt.title('Portfolio Performance')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()