from models.cointegration import CointegrationModel
import pandas as pd
from typing import Dict, List, Tuple, Union


class CointegrationTrader:
    def __init__(
        self,
        model: CointegrationModel,
        entry_threshold: float = 3.0,
        stop_loss: float = 0.05,
        profit_target: float = 0.05,
        z_window: int = 20,
    ):
        self.model = model
        self.entry_threshold = entry_threshold
        self.stop_loss = stop_loss
        self.profit_target = profit_target
        self.z_window = z_window
        self.trades = []

    def run_backtest(
        self,
        y_train: pd.Series,
        x_train: pd.Series,
        y_test: pd.Series,
        x_test: pd.Series,
    ) -> pd.DataFrame:
        self.model.fit(y_train, x_train)
        if not self.model.is_cointegrated:
            print("No cointegration detected in training period.")
            return pd.DataFrame()

        # Calculate residuals on the test period using the trained model
        test_residuals = y_test - self.model.predict(x_test)
        z_scores = (test_residuals - self.model.mu_train) / self.model.sigma_train
        open_trade = None
        equity_curve = []

        for t in range(len(z_scores)):
            z = z_scores[t]
            price_y = y_test.iloc[t]
            price_x = x_test.iloc[t]
            pnl = 0  # Initialize pnl with a default value

            if open_trade is None:
                if z > self.entry_threshold:
                    open_trade = {
                        "entry_time": t,
                        "side": "short_y_long_x",
                        "entry_price_y": price_y,
                        "entry_price_x": price_x,
                    }
                elif z < -self.entry_threshold:
                    open_trade = {
                        "entry_time": t,
                        "side": "long_y_short_x",
                        "entry_price_y": price_y,
                        "entry_price_x": price_x,
                    }

            else:
                hedge_ratio = self.model.beta
                entry_price_y = open_trade["entry_price_y"]
                entry_price_x = open_trade["entry_price_x"]

                if open_trade["side"] == "long_y_short_x":
                    pnl = (price_y - entry_price_y) - hedge_ratio * (
                        price_x - entry_price_x
                    )
                else:
                    pnl = (entry_price_y - price_y) - hedge_ratio * (
                        entry_price_x - price_x
                    )

                if abs(z) < 0.1 or pnl < -self.stop_loss or pnl > self.profit_target:
                    self.trades.append(
                        {
                            "entry_time": open_trade["entry_time"],
                            "exit_time": t,
                            "side": open_trade["side"],
                            "entry_price_y": open_trade["entry_price_y"],
                            "entry_price_x": open_trade["entry_price_x"],
                            "exit_price_y": price_y,
                            "exit_price_x": price_x,
                            "pnl": pnl,
                        }
                    )
                    open_trade = None

            equity_curve.append(pnl if open_trade else 0)
        return pd.DataFrame(self.trades)
