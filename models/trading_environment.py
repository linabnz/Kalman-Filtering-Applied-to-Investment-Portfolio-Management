# models/trading_environment.py

import numpy as np


class TradingEnvironment:
    """
    Environnement de trading modélisé comme un Processus de Décision Markovien
    selon la description dans le papier
    """
    def __init__(self, data, regression_coefs, kalman_filter=None, structural_break_model=None):
        """
        Initialise l'environnement de trading
        
        Args:
            data: DataFrame contenant les prix des instruments
            regression_coefs: Coefficients de régression pour calculer le spread
            kalman_filter: Instance du filtre de Kalman pour la décomposition du spread
            structural_break_model: Modèle pour prédire les ruptures structurelles
        """
        self.data = data
        self.regression_coefs = regression_coefs
        self.kalman_filter = kalman_filter
        self.structural_break_model = structural_break_model
        
        
        self.entry_levels = np.linspace(0.5, 2.5, 5)     
        self.stop_loss_levels = np.linspace(2.0, 5.0, 5)  
        self.exit_levels = np.linspace(0.0, 1.0, 5)      
        
       
        self.actions = []
        for entry in self.entry_levels:
            for stop_loss in self.stop_loss_levels:
                for exit in self.exit_levels:
                    self.actions.append((entry, stop_loss, exit))
        
        self.action_space = len(self.actions)
        
        
        self.current_step = 0
        self.current_positions = np.zeros(len(regression_coefs))  
        self.in_position = False
        self.last_action = None
        
        
        self.spreads = self._calculate_spreads()
        
        
        self.spread_mean = np.mean(self.spreads[:100])  
        self.spread_std = np.std(self.spreads[:100])
    
    def _calculate_spreads(self):
        """Calcule les spreads pour toutes les périodes"""
        spreads = []
        for t in range(len(self.data)):
            prices = self.data.iloc[t].values
            spread = prices[0] - self.regression_coefs[0] - np.sum(self.regression_coefs[1:] * prices[1:])
            spreads.append(spread)
        return np.array(spreads)
    
    def reset(self):
        """Réinitialise l'environnement"""
        self.current_step = 100  
        self.current_positions = np.zeros(len(self.regression_coefs))
        self.in_position = False
        self.last_action = None
        return self._get_state()
    
    def _get_state(self):
        
        
        current_spread = (self.spreads[self.current_step] - self.spread_mean) / self.spread_std
        
        
        if self.kalman_filter is not None:
            mean_reverting, random_walk = self.kalman_filter.decompose_spread(self.spreads[self.current_step])
            mean_reverting = (mean_reverting - self.spread_mean) / self.spread_std
            random_walk = (random_walk - self.spread_mean) / self.spread_std
        else:
            mean_reverting = current_spread
            random_walk = 0.0
        
        
        if self.structural_break_model is not None:
            
            window = self.spreads[self.current_step-30:self.current_step]
            structural_break_prob = self.structural_break_model.predict_proba(window.reshape(1, -1))[0][1]
        else:
            structural_break_prob = 0.0
        
       
        time_to_close = 1.0 - (self.current_step / len(self.data))
        market_close_risk = 0.1 * (1.0 - time_to_close)  
        
        
        state = [
            current_spread,
            mean_reverting,
            random_walk
        ]
        
        
        state.extend(self.current_positions)
        
        
        if self.last_action is not None:
            state.extend(self.actions[self.last_action])
        else:
            state.extend([0, 0, 0])  
        
        state.append(structural_break_prob)
        state.append(market_close_risk)
        
        return np.array(state)
    
    def step(self, action_idx):
       
        
        entry_level, stop_loss_level, exit_level = self.actions[action_idx]
        self.last_action = action_idx
        
       
        current_spread = (self.spreads[self.current_step] - self.spread_mean) / self.spread_std
        
        
        if self.kalman_filter is not None:
            mean_reverting, random_walk = self.kalman_filter.decompose_spread(self.spreads[self.current_step])
            mean_reverting = (mean_reverting - self.spread_mean) / self.spread_std
        else:
            mean_reverting = current_spread
            random_walk = 0.0
        
        
        reward = 0.0
        
        
        if not self.in_position:
            if abs(mean_reverting) > entry_level:
                
                self.in_position = True
                
                
                position_sign = -np.sign(mean_reverting)
                self.current_positions = position_sign * self.regression_coefs
                
                
                self.current_positions = self.current_positions / np.sum(np.abs(self.current_positions))
                
                
                transaction_cost = 0.001 * np.sum(np.abs(self.current_positions))
                reward -= transaction_cost
        else:
            
            if abs(mean_reverting) < exit_level:
                
                reward += np.sum(self.current_positions * (self.data.iloc[self.current_step].values - 
                                                           self.data.iloc[self.current_step-1].values))
                
                
                transaction_cost = 0.001 * np.sum(np.abs(self.current_positions))
                reward -= transaction_cost
                
                
                self.current_positions = np.zeros(len(self.regression_coefs))
                self.in_position = False
            
            
            elif abs(mean_reverting) > stop_loss_level:
                
                reward += np.sum(self.current_positions * (self.data.iloc[self.current_step].values - 
                                                           self.data.iloc[self.current_step-1].values))
                
                
                transaction_cost = 0.001 * np.sum(np.abs(self.current_positions))
                reward -= transaction_cost
                
                
                self.current_positions = np.zeros(len(self.regression_coefs))
                self.in_position = False
            
            
            else:
                reward += np.sum(self.current_positions * (self.data.iloc[self.current_step].values - 
                                                           self.data.iloc[self.current_step-1].values))
        
        
        self.current_step += 1
        
        
        done = self.current_step >= len(self.data) - 1
        
        
        if done and self.in_position:
            
            reward += np.sum(self.current_positions * (self.data.iloc[self.current_step].values - 
                                                       self.data.iloc[self.current_step-1].values))
            
            
            transaction_cost = 0.001 * np.sum(np.abs(self.current_positions))
            reward -= transaction_cost
            
            
            self.current_positions = np.zeros(len(self.regression_coefs))
            self.in_position = False
        
        
        next_state = self._get_state() if not done else None
        
        return next_state, reward, done, {}