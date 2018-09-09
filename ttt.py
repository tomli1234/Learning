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
        initial_Q[:] = -100
        self.Q = pd.DataFrame([{'state': initial_Q, 'action': (0, 0), 'Q': 0}])

    
    def move(self, g):
      
        if g.finish == False:
            valid = g.state == -100
            i, j = np.where(valid)
            self.state_t = g.state + 0 # why upadte automatically if not + 0?
    
            
            # random move
            selected = np.random.choice(len(i), 1)[0]

            #np.array_equal(game_1.state, player_1.Q['state'].iloc[0])
    
            self.move_t = i[selected], j[selected]
    
            # Change state
            g.state[self.move_t] = self.position

            # Check winner
            g.check_winner()
            self.win = g.winner[self.position]
            
            # Check finish
            g.check_finish()

    
    def update_Q(self):
        reward = np.float(self.win)
        
        Q = reward
        self.Q = self.Q.append(pd.DataFrame([{'state': self.state_t, 'action': self.move_t, 'Q': Q}]))
        print(self.Q)
        

        
        

class game():
    def __init__(self):
        self.state = np.zeros((3, 3))
        self.state[:] = -100
        self.finish = False
        self.winner = [0, 0]


    def check_winner(self):
        for i in [0, 1]:
            i_win = self.state == i
            row_sum = np.sum(i_win, axis = 1)
            col_sum = np.sum(i_win, axis = 0)
            self.winner[i] = (3 in row_sum) or (3 in col_sum)
            
    def check_finish(self):
        if sum(self.winner) > 0 or np.sum(self.state) >= 0:
            self.finish = True
        
    

player_1 = player(1)
player_0 = player(0)

for i in range(100):
    game_1 = game()
    while game_1.finish == False:
        player_1.move(game_1)
        player_0.move(game_1)
        player_1.update_Q()

        
# Evaluation
# random move baseline
player_1 = player(1)
player_0 = player(0)

win_record = []
for i in range(1000):
    game_1 = game()
    while game_1.finish == False:
        player_1.move(game_1)
        player_0.move(game_1)
    win_record.append(player_1.win)

pd.DataFrame(win_record)[0].value_counts()

        
        
        
