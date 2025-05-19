import numpy as np
import torch
import os


class DQNTradingStrategy:
    
    def __init__(self, dqn_agent, action_space):
        
        self.dqn_agent = dqn_agent
        self.action_space = action_space
    
    def decide_entry_exit_levels(self, state):
        
        action_idx = self.dqn_agent.get_action(state, epsilon=0.0)
        
        
        entry_level, stop_loss_level, exit_level = self.action_space[action_idx]
        
        return entry_level, stop_loss_level, exit_level