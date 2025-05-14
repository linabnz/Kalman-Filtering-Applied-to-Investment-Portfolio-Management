from strategies.cointegrationtrading import CointegrationTrader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_trades(trader: CointegrationTrader, y: pd.Series, x: pd.Series):
    model = trader.model
    spread = model.residuals
    z_scores = model.z_scores
    trades = pd.DataFrame(trader.trades)

    window = trader.z_window
    mean = pd.Series(spread).rolling(window).mean()
    std = pd.Series(spread).rolling(window).std()
    upper = mean + trader.entry_threshold * std
    lower = mean - trader.entry_threshold * std

    fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # 1. Log-prices
    axs[0].plot(np.log(y), label="log(y)")
    axs[0].plot(np.log(x), label="log(x)")
    axs[0].set_title("Log Prices")
    axs[0].legend()

    # 2. Spread and thresholds
    axs[1].plot(spread, label="Spread (Residual)")
    axs[1].plot(upper, "r--", label="Upper Threshold")
    axs[1].plot(lower, "r--", label="Lower Threshold")
    axs[1].axhline(0, color="black", linestyle=":")
    axs[1].set_title("Spread and Entry/Exit Thresholds")
    axs[1].legend()

    for _, row in trades.iterrows():
        axs[1].axvline(row["entry_time"], color="blue", linestyle="--", alpha=0.6)
        axs[1].axvline(row["exit_time"], color="red", linestyle="--", alpha=0.6)

    # 3. Cumulative PnL
    cum_pnl = (
        trades.set_index("exit_time")["pnl"]
        .reindex(range(len(z_scores)))
        .fillna(0)
        .cumsum()
    )
    axs[2].plot(cum_pnl, label="Cumulative PnL")
    axs[2].set_title("Cumulative Strategy Return")
    axs[2].legend()

    plt.tight_layout()
    plt.show()
