import pandas as pd
from typing import List, Optional, Tuple, Union
from abc import ABC, abstractmethod

import torch


class BaseCointegrationTrader(ABC):
    def __init__(
        self,
        model,
        entry_threshold: float,
        stop_loss: float,
        profit_target: float,
        rolling_window: int,
    ):
        self.model = model
        self.entry_threshold = entry_threshold
        self.stop_loss = stop_loss
        self.profit_target = profit_target
        self.rolling_window = rolling_window
        self.trades: List[dict] = []

    def run_backtest(self, y: pd.Series, x: pd.Series) -> pd.DataFrame:
        trades_log = []
        open_trade = None
        frozen_model = None

        for t in range(self.rolling_window, len(y)):
            date = y.index[t]
            y_train = y.iloc[t - self.rolling_window : t]
            x_train = x.iloc[t - self.rolling_window : t]
            y_t, x_t = y.iloc[t], x.iloc[t]

            if open_trade is None:
                self.model.fit(y_train, x_train)
                if not self.model.is_cointegrated:
                    continue

                signal = self.get_trade_signal(y_t, x_t)
                if signal is None:
                    continue

                if isinstance(signal, dict):
                    z = signal.get("z")
                    extra = {
                        "mr_component": signal.get("mr_component", 0),
                        "rw_component": signal.get("rw_component", 0),
                        "break_probability": signal.get("break_probability", 0),
                    }
                else:
                    z, extra = signal
                if z is None:
                    continue

                extra = {
                    "mr_component": extra.get("mr_component", 0),
                    "rw_component": extra.get("rw_component", 0),
                    "break_probability": extra.get("break_probability", 0),
                }

                if z > self.entry_threshold:
                    open_trade = self._build_trade("short_y_long_x", y_t, x_t, z, extra)
                elif z < -self.entry_threshold:
                    open_trade = self._build_trade("long_y_short_x", y_t, x_t, z, extra)
                else:
                    continue

                frozen_model = self.model
                trades_log.append(self._log_trade(date, z, open_trade, 0))
            else:
                z, pnl = self.update_trade(y_t, x_t, frozen_model, open_trade)
                trades_log.append(self._log_trade(date, z, open_trade, pnl))

                if abs(z) < 0.1 or pnl < -self.stop_loss or pnl > self.profit_target:
                    self.trades.append(
                        {
                            **open_trade,
                            "exit_time": date,
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
                        self._log_trade(date, z, open_trade, pnl, close=True)
                    )
                    open_trade = None
                    frozen_model = None

        return pd.DataFrame(trades_log)

    def _build_trade(self, side, y_price, x_price, z, extra=None):
        trade = {
            "entry_time": None,
            "side": side,
            "entry_price_y": y_price,
            "entry_price_x": x_price,
            "entry_z": z,
            "entry_mr": extra.get("mr_component", 0) if extra else 0,  # Add entry_mr
        }
        if extra:
            trade.update(extra)
        return trade

    def _log_trade(self, date, z, trade, pnl, close=False):
        pos = 0 if close else (-1 if trade["side"] == "long_y_short_x" else 1)
        log = {"date": date, "position": pos, "z_score": z, "pnl": pnl}
        if "mr_component" in trade:
            log["mr_component"] = trade["mr_component"]
            log["rw_component"] = trade["rw_component"]
        if "break_probability" in trade:
            log["break_probability"] = trade["break_probability"]
        return log

    @abstractmethod
    def get_trade_signal(self, y_t: float, x_t: float) -> Optional[Tuple[float, dict]]:
        pass

    @abstractmethod
    def update_trade(
        self, y_t: float, x_t: float, model, trade: dict
    ) -> Tuple[float, float]:
        pass

    def get_performance_metrics(self) -> dict:
        if not self.trades:
            return dict(
                total_trades=0, win_rate=0, avg_pnl=0, total_pnl=0, sharpe_ratio=0
            )

        df = pd.DataFrame(self.trades)
        total = len(df)
        wins = (df["pnl"] > 0).sum()
        return {
            "total_trades": total,
            "win_rate": wins / total,
            "avg_pnl": df["pnl"].mean(),
            "total_pnl": df["pnl"].sum(),
            "sharpe_ratio": (
                df["pnl"].mean() / df["pnl"].std() if df["pnl"].std() > 0 else 0
            ),
            "exit_reasons": (
                df["exit_reason"].value_counts().to_dict()
                if "exit_reason" in df
                else {}
            ),
        }


class SimpleCointegrationTrader(BaseCointegrationTrader):
    def __init__(
        self,
        model,
        entry_threshold=2.0,
        stop_loss=0.05,
        profit_target=0.05,
        rolling_window=60,
    ):
        super().__init__(
            model, entry_threshold, stop_loss, profit_target, rolling_window
        )

    def get_trade_signal(self, y_t, x_t):
        spread = y_t - (self.model.beta * x_t + self.model.intercept)
        if self.model.sigma == 0:
            return None
        z = (spread - self.model.mu) / self.model.sigma
        return (z, {})

    def update_trade(self, y_t, x_t, model, trade):
        spread = y_t - (model.beta * x_t + model.intercept)
        z = (spread - model.mu) / model.sigma
        pnl = (
            (
                (y_t - trade["entry_price_y"])
                - model.beta * (x_t - trade["entry_price_x"])
            )
            if trade["side"] == "long_y_short_x"
            else (
                (trade["entry_price_y"] - y_t)
                - model.beta * (trade["entry_price_x"] - x_t)
            )
        )
        return z, pnl


class PartialCointegrationTrader(BaseCointegrationTrader):
    def __init__(
        self,
        model,
        entry_threshold=1.25,
        stop_loss=0.05,
        profit_target=0.05,
        rolling_window=60,
        kalman_gain=0.7,
        structural_break_model=None,
        input_length=300,
    ):
        super().__init__(
            model, entry_threshold, stop_loss, profit_target, rolling_window
        )
        self.kalman_gain = kalman_gain
        self.structural_break_model = structural_break_model
        self.input_length = input_length

    def get_trade_signal(self, y_t, x_t):
        break_prob = None

        if self.structural_break_model:
            spread_window = self.model.residuals[-self.input_length :]
            if len(spread_window) < self.input_length:
                return None

            spread_tensor = torch.tensor(spread_window, dtype=torch.float32).unsqueeze(
                0
            )
            y_tensor = torch.tensor(
                [y_t] * self.input_length, dtype=torch.float32
            ).unsqueeze(0)
            x_tensor = torch.tensor(
                [x_t] * self.input_length, dtype=torch.float32
            ).unsqueeze(0)

            break_prob = self.structural_break_model(
                spread_tensor, y_tensor, x_tensor
            ).item()

            if break_prob > 0.5:
                return {
                    "z": None,
                    "mr_component": None,
                    "rw_component": None,
                    "break_probability": break_prob,
                }

        spread = y_t - (self.model.beta * x_t + self.model.intercept)
        last_mr = self.model.mean_reverting_component[-1]
        last_rw = self.model.random_walk_component[-1]

        mr_pred = self.model.rho * last_mr
        rw_pred = last_rw
        innovation = spread - (mr_pred + rw_pred)

        mr_t = mr_pred + self.kalman_gain * innovation
        rw_t = rw_pred + (1 - self.kalman_gain) * innovation

        std = (
            pd.Series(self.model.mean_reverting_component)
            .rolling(window=20)
            .std()
            .iloc[-1]
        )
        if std == 0 or pd.isna(std):
            return None

        z = mr_t / std
        return {
            "z": z,
            "mr_component": mr_t,
            "rw_component": rw_t,
            "break_probability": break_prob,
        }

    def update_trade(self, y_t, x_t, model, trade):
        spread = y_t - (model.beta * x_t + model.intercept)
        last_mr = trade["mr_component"]
        last_rw = trade["rw_component"]

        mr_pred = model.rho * last_mr
        rw_pred = last_rw
        innovation = spread - (mr_pred + rw_pred)

        mr_t = mr_pred + model.kalman_gain * innovation
        rw_t = rw_pred + (1 - model.kalman_gain) * innovation

        std = abs(trade["entry_mr"] / trade["entry_z"])
        z = mr_t / std if std != 0 else 0

        pnl = (
            (
                (y_t - trade["entry_price_y"])
                - model.beta * (x_t - trade["entry_price_x"])
            )
            if trade["side"] == "long_y_short_x"
            else (
                (trade["entry_price_y"] - y_t)
                - model.beta * (trade["entry_price_x"] - x_t)
            )
        )

        trade["mr_component"] = mr_t
        trade["rw_component"] = rw_t
        return z, pnl
