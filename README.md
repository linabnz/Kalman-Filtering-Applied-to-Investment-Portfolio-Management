# Kalman Filtering Applied to Investment Portfolio Management

This project explores the application of Kalman filtering and other advanced statistical techniques to optimize investment portfolio management. It integrates methods like cointegration analysis, reinforcement learning, and structural break detection to develop robust trading strategies.

## Features

- **Cointegration Analysis**: Identify pairs of assets with long-term equilibrium relationships.
- **Kalman Filtering**: Dynamically estimate parameters for trading strategies.
- **Reinforcement Learning**: Optimize trading decisions using Q-learning.
- **Structural Break Detection**: Adapt strategies to market regime changes.
- **Data Visualization**: Generate insightful plots for analysis and reporting.

## Project Structure

- **`data/`**: Contains datasets like stock prices, sectors, and weekly returns.
- **`models/`**: Includes implementations of Kalman filters, cointegration models, and Q-networks.
- **`results/`**: Stores analysis outputs, such as cointegration results and trading performance plots.
- **`sandbox/`**: Jupyter notebooks for exploratory data analysis and testing.
- **`scripts_for_main/`**: Scripts for running specific analyses or strategies.
- **`strategies/`**: Modules for implementing trading strategies.
- **`utils/`**: Utility functions for data loading, statistical tests, and visualization.

## Getting Started

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Kalman-Filtering-Applied-to-Investment-Portfolio-Management
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the main script:
   ```bash
   python main.py
   ```