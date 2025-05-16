from typing import List
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import numpy as np
from models.cointegration import CointegrationModel


class CointegrationTrader:
    def __init__(
        self,
        model: CointegrationModel,
        entry_threshold: float = 2.0,
        stop_loss: float = 0.05,
        profit_target: float = 0.05,
        rolling_window: int = 60,
    ):
        self.model = model
        self.entry_threshold = entry_threshold
        self.stop_loss = stop_loss
        self.profit_target = profit_target
        self.rolling_window = rolling_window
        self.trades: List[dict] = []

    def run_backtest(
        self,
        y: pd.Series,
        x: pd.Series,
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
                self.model.fit(y_train, x_train)

                if not self.model.is_cointegrated:
                    continue

                spread_t = y_t - (self.model.beta * x_t + self.model.intercept)
                z = (spread_t - self.model.mu) / self.model.sigma

                if self.model.sigma == 0 or pd.isna(z):
                    continue

                if z > self.entry_threshold:
                    open_trade = {
                        "entry_time": date,
                        "side": "short_y_long_x",
                        "entry_price_y": y_t,
                        "entry_price_x": x_t,
                        "entry_z": z,
                    }
                    frozen_model = self.model
                    trades_log.append(
                        {"date": date, "position": 1, "z_score": z, "pnl": 0}
                    )
                elif z < -self.entry_threshold:
                    open_trade = {
                        "entry_time": date,
                        "side": "long_y_short_x",
                        "entry_price_y": y_t,
                        "entry_price_x": x_t,
                        "entry_z": z,
                    }
                    frozen_model = self.model
                    trades_log.append(
                        {"date": date, "position": -1, "z_score": z, "pnl": 0}
                    )
            else:
                hedge_ratio = frozen_model.beta
                intercept = frozen_model.intercept
                mu = frozen_model.mu
                sigma = frozen_model.sigma

                spread_t = y_t - (hedge_ratio * x_t + intercept)
                z = (spread_t - mu) / sigma

                entry_price_y = open_trade["entry_price_y"]
                entry_price_x = open_trade["entry_price_x"]

                if open_trade["side"] == "long_y_short_x":
                    pnl = (y_t - entry_price_y) - hedge_ratio * (x_t - entry_price_x)
                else:
                    pnl = (entry_price_y - y_t) - hedge_ratio * (entry_price_x - x_t)

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
                        }
                    )
                    trades_log.append(
                        {"date": date, "position": 0, "z_score": z, "pnl": pnl}
                    )
                    open_trade = None
                    frozen_model = None

        return pd.DataFrame(trades_log)
