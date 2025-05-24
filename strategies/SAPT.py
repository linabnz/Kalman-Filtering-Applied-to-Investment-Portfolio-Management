from typing import List, Tuple
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import numpy as np
import torch
from models.cointegrationModel import PartialCointegrationModel


class PartialCointegrationTraderSAPT:
    def __init__(
        self,
        model: PartialCointegrationModel,
        entry_threshold: float = 1.25,
        stop_loss: float = 0.05,
        profit_target: float = 0.05,
        rolling_window: int = 60,
        kalman_gain: float = 0.7,
    ):
        self.model = model
        self.entry_threshold = entry_threshold
        self.stop_loss = stop_loss
        self.profit_target = profit_target
        self.rolling_window = rolling_window
        self.kalman_gain = kalman_gain
        self.trades: List[dict] = []

    def run_backtest(
        self, y: pd.Series, x: pd.Series, structural_break_model=None, input_length=300
    ) -> pd.DataFrame:
        trades_log = []
        open_trade = None
        frozen_model = None

        for t in range(self.rolling_window, len(y)):
            date = y.index[t]
            y_train = y.iloc[t - self.rolling_window : t]
            x_train = x.iloc[t - self.rolling_window : t]
            y_t = y.iloc[t]
            x_t = x.iloc[t]

            if open_trade is None:
                # Fit model to training data
                self.model.fit(y_train, x_train)

                if not self.model.is_cointegrated:
                    continue

                if structural_break_model:
                    # Extraire fenêtre temporelle
                    spread_window = y_train.values - (
                        self.model.beta * x_train.values + self.model.intercept
                    )
                    y_window = y_train.values
                    x_window = x_train.values

                    # Conversion en tensors torch
                    spread_tensor = torch.tensor(
                        spread_window[-input_length:], dtype=torch.float32
                    ).unsqueeze(0)
                    y_tensor = torch.tensor(
                        y_window[-input_length:], dtype=torch.float32
                    ).unsqueeze(0)
                    x_tensor = torch.tensor(
                        x_window[-input_length:], dtype=torch.float32
                    ).unsqueeze(0)

                    break_prob = structural_break_model(
                        spread_tensor, y_tensor, x_tensor
                    ).item()

                    # Seuil à ajuster (ex. 0.5 ou calibré empiriquement)
                    if break_prob > 0.5:
                        continue  # on n'entre pas en position

                # Use mean-reverting component for signal generation
                # We calculate the latest spread
                spread_t = y_t - (self.model.beta * x_t + self.model.intercept)

                # Predict the mean-reverting and random walk components for the new spread
                # We need to manually update the components based on the last known states
                mr_pred = self.model.rho * self.model.mean_reverting_component[-1]
                rw_pred = self.model.random_walk_component[-1]

                # Calculate innovation (prediction error)
                innovation = spread_t - (mr_pred + rw_pred)

                # Update components using Kalman filter
                mr_t = mr_pred + self.model.kalman_gain * innovation
                rw_t = rw_pred + (1 - self.model.kalman_gain) * innovation

                # Calculate z-score for mean-reverting component
                mr_series = pd.Series(self.model.mean_reverting_component)
                mr_rolling_std = mr_series.rolling(window=20).std().iloc[-1]
                if mr_rolling_std == 0 or pd.isna(mr_rolling_std):
                    continue

                # Use mean-reverting component's z-score for trading signals
                z = mr_t / mr_rolling_std

                if pd.isna(z):
                    continue

                if z > self.entry_threshold:
                    open_trade = {
                        "entry_time": date,
                        "side": "short_y_long_x",
                        "entry_price_y": y_t,
                        "entry_price_x": x_t,
                        "entry_z": z,
                        "entry_mr": mr_t,
                        "entry_rw": rw_t,
                    }
                    frozen_model = self.model
                    trades_log.append(
                        {
                            "date": date,
                            "position": 1,
                            "z_score": z,
                            "mr_component": mr_t,
                            "rw_component": rw_t,
                            "pnl": 0,
                        }
                    )
                elif z < -self.entry_threshold:
                    open_trade = {
                        "entry_time": date,
                        "side": "long_y_short_x",
                        "entry_price_y": y_t,
                        "entry_price_x": x_t,
                        "entry_z": z,
                        "entry_mr": mr_t,
                        "entry_rw": rw_t,
                    }
                    frozen_model = self.model
                    trades_log.append(
                        {
                            "date": date,
                            "position": -1,
                            "z_score": z,
                            "mr_component": mr_t,
                            "rw_component": rw_t,
                            "pnl": 0,
                        }
                    )
            else:
                # When a trade is open, we continue tracking both components
                hedge_ratio = frozen_model.beta
                intercept = frozen_model.intercept

                # Calculate current spread
                spread_t = y_t - (hedge_ratio * x_t + intercept)

                # Update mean-reverting and random walk components
                last_mr = trades_log[-1]["mr_component"]
                last_rw = trades_log[-1]["rw_component"]

                mr_pred = frozen_model.rho * last_mr
                rw_pred = last_rw

                innovation = spread_t - (mr_pred + rw_pred)

                mr_t = mr_pred + frozen_model.kalman_gain * innovation
                rw_t = rw_pred + (1 - frozen_model.kalman_gain) * innovation

                # Calculate z-score for mean reverting component
                # We use the standard deviation from when we entered the trade
                mr_std = abs(open_trade["entry_mr"] / open_trade["entry_z"])
                z = mr_t / mr_std if mr_std != 0 else 0

                entry_price_y = open_trade["entry_price_y"]
                entry_price_x = open_trade["entry_price_x"]

                # Calculate PnL - both components affect PnL, but we exit based on mean-reverting component
                if open_trade["side"] == "long_y_short_x":
                    pnl = (y_t - entry_price_y) - hedge_ratio * (x_t - entry_price_x)
                else:
                    pnl = (entry_price_y - y_t) - hedge_ratio * (entry_price_x - x_t)

                # Track the trade
                trades_log.append(
                    {
                        "date": date,
                        "position": -1 if open_trade["side"] == "long_y_short_x" else 1,
                        "z_score": z,
                        "mr_component": mr_t,
                        "rw_component": rw_t,
                        "pnl": pnl,
                    }
                )

                # Exit condition: mean-reverting component crosses zero,
                # or stop loss/profit target hit
                if abs(z) < 0.1 or pnl < -self.stop_loss or pnl > self.profit_target:
                    self.trades.append(
                        {
                            "entry_time": open_trade["entry_time"],
                            "exit_time": date,
                            "side": open_trade["side"],
                            "entry_price_y": entry_price_y,
                            "entry_price_x": entry_price_x,
                            "exit_price_y": y_t,
                            "exit_price_x": x_t,
                            "pnl": pnl,
                            "exit_reason": (
                                "zero_crossing"
                                if abs(z) < 0.1
                                else (
                                    "stop_loss"
                                    if pnl < -self.stop_loss
                                    else "profit_target"
                                )
                            ),
                        }
                    )
                    trades_log.append(
                        {
                            "date": date,
                            "position": 0,
                            "z_score": z,
                            "mr_component": mr_t,
                            "rw_component": rw_t,
                            "pnl": pnl,
                        }
                    )
                    open_trade = None
                    frozen_model = None

        return pd.DataFrame(trades_log)

    def get_performance_metrics(self) -> dict:
        """
        Calculate performance metrics for the backtest.

        Returns:
            Dictionary of performance metrics
        """
        if not self.trades:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "avg_pnl": 0,
                "total_pnl": 0,
                "sharpe_ratio": 0,
            }

        trades_df = pd.DataFrame(self.trades)

        # Calculate basic metrics
        total_trades = len(trades_df)
        profitable_trades = len(trades_df[trades_df["pnl"] > 0])
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        avg_pnl = trades_df["pnl"].mean()
        total_pnl = trades_df["pnl"].sum()

        # Calculate Sharpe ratio (if there are more than 1 trade)
        if len(trades_df) > 1:
            sharpe_ratio = (
                trades_df["pnl"].mean() / trades_df["pnl"].std()
                if trades_df["pnl"].std() > 0
                else 0
            )
        else:
            sharpe_ratio = 0

        # Count exit reasons
        exit_reasons = (
            trades_df["exit_reason"].value_counts().to_dict()
            if "exit_reason" in trades_df.columns
            else {}
        )

        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "avg_pnl": avg_pnl,
            "total_pnl": total_pnl,
            "sharpe_ratio": sharpe_ratio,
            "exit_reasons": exit_reasons,
        }
