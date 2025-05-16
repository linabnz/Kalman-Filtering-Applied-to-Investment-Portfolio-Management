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


def plot_cointegration_trade(
    prices, ticker1, ticker2, trades, beta, entry_threshold=2.0
):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Subplot 1: Log Prices
    log_price1 = np.log(prices[ticker1])
    log_price2 = np.log(prices[ticker2])
    axes[0].plot(log_price1, label=f"{ticker1}")
    axes[0].plot(log_price2, label=f"{ticker2}")
    axes[0].set_title("Closing Prices")
    axes[0].set_ylabel("Log Price")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Subplot 2: Spread (Error signal)
    spread = log_price1 - beta * log_price2
    rolling_mean = spread.rolling(60).mean()
    rolling_std = spread.rolling(60).std()
    upper = rolling_mean + entry_threshold * rolling_std
    lower = rolling_mean - entry_threshold * rolling_std

    axes[1].plot(spread, label="Sector Error", color="olive")
    axes[1].plot(upper, label="Upper Threshold", linestyle="--", color="red")
    axes[1].plot(lower, label="Lower Threshold", linestyle="--", color="blue")

    if not trades.empty:
        entries = trades[trades["position"] != 0]
        exits = trades[trades["position"] == 0]

        axes[1].scatter(
            entries["date"],
            spread.loc[entries["date"]],
            color="blue",
            marker="^",
            label="Entry Point",
        )
        axes[1].scatter(
            exits["date"],
            spread.loc[exits["date"]],
            color="red",
            marker="v",
            label="Exit Point",
        )

    axes[1].set_title("Sector Error and Entry/Exit Points")
    axes[1].set_ylabel("Sector Error")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Subplot 3: Cumulative Transaction Return
    transaction_return = (
        trades.set_index("date")["pnl"].cumsum()
        if not trades.empty
        else pd.Series(dtype=float)
    )
    axes[2].plot(transaction_return, label="Transaction Return", color="slateblue")

    if not trades.empty:
        axes[2].scatter(
            entries["date"],
            transaction_return.loc[entries["date"]],
            color="blue",
            marker="^",
            label="Entry Point",
        )
        axes[2].scatter(
            exits["date"],
            transaction_return.loc[exits["date"]],
            color="red",
            marker="v",
            label="Exit Point",
        )

    axes[2].axhline(y=0, color="black", linestyle="--")
    axes[2].set_title(
        f"Transaction Return; Return = {transaction_return.iloc[-1]:.2%} ; Duration = {len(transaction_return)} days"
        if not transaction_return.empty
        else "Transaction Return"
    )
    axes[2].set_ylabel("Transaction Return (%)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.xlabel("Days")
    plt.tight_layout()
    plt.show()
