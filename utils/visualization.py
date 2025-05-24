from strategies.cointegrationTrader import SimpleCointegrationTrader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_trades(trader: SimpleCointegrationTrader, y: pd.Series, x: pd.Series):
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
        # Ensure transaction_return index has no duplicates
        transaction_return = transaction_return[
            ~transaction_return.index.duplicated(keep="first")
        ]

        # Align entries and transaction_return
        valid_entries = entries["date"].drop_duplicates().isin(transaction_return.index)
        aligned_entries = entries.loc[valid_entries]

        axes[2].scatter(
            aligned_entries["date"],
            transaction_return.reindex(aligned_entries["date"]).values,
            color="blue",
            marker="^",
            label="Entry Point",
        )

        valid_exits = exits["date"].drop_duplicates().isin(transaction_return.index)
        aligned_exits = exits.loc[valid_exits]

        axes[2].scatter(
            aligned_exits["date"],
            transaction_return.reindex(aligned_exits["date"]).values,
            color="red",
            marker="v",
            label="Exit Point",
        )

    axes[2].axhline(y=0, color="black", linestyle="--")
    axes[2].set_title(
        f"Duration = {len(transaction_return)} days"
        if not transaction_return.empty
        else "Transaction Return"
    )
    axes[2].set_ylabel("Transaction Return")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.xlabel("Days")
    plt.tight_layout()
    plt.show()


def plot_partial_cointegration_trade(
    prices: pd.DataFrame,
    ticker1: str,
    ticker2: str,
    trades: pd.DataFrame,
    beta: float,
    entry_threshold: float = 1.25,
    mean_reverting: np.ndarray = None,
    random_walk: np.ndarray = None,
):
    """
    Plot partial cointegration trading results including mean-reverting and random walk components.

    Args:
        prices: DataFrame with price data
        ticker1: First ticker symbol
        ticker2: Second ticker symbol
        trades: DataFrame with trade data
        beta: Regression coefficient (hedge ratio)
        entry_threshold: Z-score threshold for entering trades
        mean_reverting: Mean-reverting component array
        random_walk: Random walk component array
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    # Create figure with 4 subplots
    fig, axs = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

    # Plot prices
    axs[0].plot(prices[ticker1], label=ticker1)
    axs[0].plot(prices[ticker2], label=ticker2)
    axs[0].set_title(f"Prices: {ticker1} vs {ticker2} (beta={beta:.4f})")
    axs[0].legend()
    axs[0].grid(True)

    # Plot positions
    if not trades.empty:
        positions = trades[trades["position"] != 0]

        # Plot entry/exit points on price chart
        for idx, pos in positions.iterrows():
            date = pos["date"]
            if abs(pos["position"]) > 0:  # Entry point
                axs[0].scatter(
                    date,
                    prices.loc[date, ticker1],
                    color="green" if pos["position"] < 0 else "red",
                    marker="^" if pos["position"] < 0 else "v",
                    s=100,
                )
                axs[0].scatter(
                    date,
                    prices.loc[date, ticker2],
                    color="red" if pos["position"] < 0 else "green",
                    marker="v" if pos["position"] < 0 else "^",
                    s=100,
                )

    # Plot mean-reverting and random walk components
    if mean_reverting is not None and random_walk is not None:
        dates = prices.index[-len(mean_reverting) :]

        axs[1].plot(dates, mean_reverting, label="Mean-Reverting", color="blue")
        axs[1].axhline(y=0, color="black", linestyle="-", alpha=0.3)
        axs[1].axhline(y=entry_threshold, color="red", linestyle="--", alpha=0.5)
        axs[1].axhline(y=-entry_threshold, color="red", linestyle="--", alpha=0.5)
        axs[1].set_title("Mean-Reverting Component")
        axs[1].legend()
        axs[1].grid(True)

        axs[2].plot(dates, random_walk, label="Random Walk", color="green")
        axs[2].set_title("Random Walk Component")
        axs[2].legend()
        axs[2].grid(True)

    # Plot PnL
    if not trades.empty:
        axs[3].plot(trades["date"], trades["pnl"], label="PnL", color="purple")
        axs[3].set_title("Trade PnL")
        axs[3].legend()
        axs[3].grid(True)

    # Format x-axis
    plt.xlabel("Date")
    plt.tight_layout()

    return fig, axs


def plot_sapt_trade(
    prices: pd.DataFrame,
    ticker1: str,
    ticker2: str,
    trades: pd.DataFrame,
    beta: float,
    entry_threshold: float = 1.25,
    mean_reverting: np.ndarray = None,
    random_walk: np.ndarray = None,
    break_probabilities: np.ndarray = None,
    break_threshold: float = 0.5,
):
    """
    Plot SAPT trading results with structural break prediction overlay.

    Args:
        prices: DataFrame with price data.
        ticker1: First ticker symbol.
        ticker2: Second ticker symbol.
        trades: DataFrame with trade data.
        beta: Hedge ratio.
        entry_threshold: Threshold for mean-reverting signal.
        mean_reverting: Array of mean-reverting component.
        random_walk: Array of random walk component.
        break_probabilities: Array of predicted structural break probabilities.
        break_threshold: Probability threshold above which trades are avoided.
    """
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(5, 1, figsize=(15, 14), sharex=True)

    # 1. Price Plot
    axs[0].plot(prices[ticker1], label=ticker1)
    axs[0].plot(prices[ticker2], label=ticker2)
    axs[0].set_title(f"Prices: {ticker1} vs {ticker2} (beta={beta:.4f})")
    axs[0].legend()
    axs[0].grid(True)

    # 2. Trade Points
    if not trades.empty:
        for _, trade in trades.iterrows():
            date = trade["date"]
            color = "green" if trade["position"] < 0 else "red"
            axs[0].scatter(
                date, prices.loc[date, ticker1], color=color, marker="^", s=100
            )
            axs[0].scatter(
                date, prices.loc[date, ticker2], color=color, marker="v", s=100
            )

    # 3. Mean-Reverting Component
    if mean_reverting is not None:
        dates = prices.index[-len(mean_reverting) :]
        axs[1].plot(dates, mean_reverting, label="Mean-Reverting", color="blue")
        axs[1].axhline(y=entry_threshold, color="red", linestyle="--", alpha=0.5)
        axs[1].axhline(y=-entry_threshold, color="red", linestyle="--", alpha=0.5)
        axs[1].set_title("Mean-Reverting Component")
        axs[1].legend()
        axs[1].grid(True)

    # 4. Random Walk Component
    if random_walk is not None:
        axs[2].plot(dates, random_walk, label="Random Walk", color="green")
        axs[2].set_title("Random Walk Component")
        axs[2].legend()
        axs[2].grid(True)

    # 5. Break Probability
    if break_probabilities is not None:
        axs[3].plot(
            dates, break_probabilities, label="Break Probability", color="orange"
        )
        axs[3].axhline(y=break_threshold, color="black", linestyle="--", alpha=0.5)
        axs[3].set_title("Predicted Structural Break Probability")
        axs[3].legend()
        axs[3].grid(True)

    # 6. PnL
    if not trades.empty:
        axs[4].plot(trades["date"], trades["pnl"], label="PnL", color="purple")
        axs[4].set_title("Trade PnL")
        axs[4].legend()
        axs[4].grid(True)

    plt.xlabel("Date")
    plt.tight_layout()
    return fig, axs
