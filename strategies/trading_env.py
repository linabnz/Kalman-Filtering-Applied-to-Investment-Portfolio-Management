import numpy as np

class TradingEnvironment:
    def __init__(self, df_spread):
        if 'spread' not in df_spread.columns:
            raise ValueError("La colonne 'spread' est manquante dans le DataFrame fourni. Normalisez d'abord les spreads.")

        self.df = df_spread.copy().reset_index(drop=True)
        self.df['temps'] = np.linspace(0, 1, len(self.df))
        self.df['position'] = 0
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        self.position = 0
        self.entry_price = None
        return self._get_state()

    def _get_state(self):
        spread = self.df['spread'].iloc[self.current_step]
        time_ratio = self.current_step / len(self.df)
        return np.array([spread, self.position, time_ratio], dtype=np.float32)

    def step(self, action, transaction_cost=100):
        spread_t = self.df['spread'].iloc[self.current_step]
        spread_next = self.df['spread'].iloc[self.current_step + 1] if self.current_step + 1 < len(self.df) else spread_t

        reward = 0
        done = False

        if action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = spread_t
            reward -= transaction_cost

        elif action == 2 and self.position == 1:
            pnl = spread_next - self.entry_price
            reward += pnl - transaction_cost
            self.position = 0
            self.entry_price = None

        elif action == 0 and self.position == 1:
            reward += (spread_next - spread_t)

        if action == 0 and self.position == 0:
            reward -= 1

        reward = np.clip(reward, -100, 100)

        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            done = True

        next_state = self._get_state()
        return next_state, reward, done