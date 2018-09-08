# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 09:00:55 2018

@author: Tom Li
"""

import numpy as np
import pandas as pd

class player():
    def __init__(self, position):
        self.position = position
        initial_Q = np.zeros((3, 3))
        initial_Q[:] = np.nan
        self.Q = pd.DataFrame([{'state': initial_Q, 'action': (0, 0), 'Q': 0.5}])

    
    def move(self, g):
        valid = np.isnan(g.state)
        i, j = np.where(valid)
        self.state_t = g.state + 0 # why upadte automatically if not + 0?

        
        # random move
        selected = np.random.choice(len(i), 1)[0]

        self.move_t = i[selected], j[selected]

        # Change state
        g.state[self.move_t] = self.position

        # Check winner
        g.check_winner()
        self.win = g.winner
        
        # Check finish
        g.check_finish()
    
    def update_Q(self):
        reward = np.float(self.win[self.position])
        
        Q = reward
        print([self.state_t, self.move_t, Q])
        

        
        

class game():
    def __init__(self):
        self.state = np.zeros((3, 3))
        self.state[:] = np.nan

    def check_winner(self):
        self.winner = [0, 0]
        for i in [0, 1]:
            i_win = self.state == i
            row_sum = np.sum(i_win, axis = 1)
            col_sum = np.sum(i_win, axis = 0)
            self.winner[i] = (3 in row_sum) or (3 in col_sum)
            
    def check_finish(self):
        self.finish = False
        if sum(self.winner) > 0 or np.sum(np.isnan(self.state)) == 0:
            self.finish = True
        
    

game_1 = game()
player_1 = player(1)
player_0 = player(0)

player_1.move(game_1)
player_0.move(game_1)
player_1.update_Q()

print(game_1.state)

game_1.winner
game_1.finish


