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

    def step(self, action, transaction_cost=10):  # Plus réaliste que 0.3 mais pas 50
        spread_t = self.df['spread'].iloc[self.current_step]
        spread_next = self.df['spread'].iloc[self.current_step + 1] if self.current_step + 1 < len(self.df) else spread_t

        reward = 0
        done = False

        # === ACTIONS DE TRADING ===
        if action == 1 and self.position == 0:  # Enter Long
            self.position = 1
            self.entry_price = spread_t
            reward -= transaction_cost  # Coût d'entrée réaliste

        elif action == 2 and self.position == 1:  # Exit
            pnl = spread_next - self.entry_price
            reward += pnl - transaction_cost  # PnL moins coût de sortie
            
            # Bonus modeste pour trade profitable (réaliste)
            if pnl > 0:
                reward += 5  # Au lieu de 35 - plus mesuré
            
            self.position = 0
            self.entry_price = None

        elif action == 0 and self.position == 1:  # Hold en position
            # Gain/perte proportionnel au mouvement du spread
            reward += (spread_next - spread_t) * 1.2  # Léger multiplier au lieu de 2.0

        # === INCITATIONS ÉQUILIBRÉES ===
        
        # Petit bonus pour maintenir une position (encourage l'engagement)
        if self.position == 1:
            reward += 0.5  # Au lieu de 1.5 - plus mesuré
        
        # Légère pénalité pour inactivité excessive (évite le Hold sans position)
        elif action == 0 and self.position == 0:
            reward -= 0.2  # Douce pénalité au lieu de -1
        
        # === CONTRAINTES RÉALISTES ===
        
        # Plafonnement plus réaliste mais pas trop restrictif
        reward = np.clip(reward, -150, 200)  # Entre ton ancien (-100,100) et nouveau (-100,600)

        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            done = True

        next_state = self._get_state()
        return next_state, reward, done
