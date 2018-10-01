# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 09:00:55 2018

@author: Tom Li
"""

import numpy as np
import pandas as pd
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', 800)

class player():
    def __init__(self, position):
        self.position = position
        initial_Q = np.zeros((3, 3))
        initial_Q[:] = -100
        self.Q = pd.DataFrame([{'state': initial_Q, 'action': (0, 0), 'Q': 0}])

    
    def move(self, g, rand_prob):
      
        if g.finish == False:
            valid = g.state == -100
            i, j = np.where(valid)
            self.state_t = g.state + 0 # why update automatically if not + 0?
    
            
            self.seen_state = self.Q['state'].apply(lambda x: np.array_equal(x, g.state))
            random = np.random.rand() < rand_prob
            
            if (random == True) or (sum(self.seen_state) == 0):
                # random move
                selected = np.random.choice(len(i), 1)[0]    
                self.move_t = i[selected], j[selected]
                #self.move_t = np.random.choice(range(3), 1)[0], np.random.choice(range(3), 1)[0]
            else:
                self.select_move = np.argmax(self.Q[self.seen_state]['Q'])
                self.move_t = self.Q[self.seen_state].iloc[self.select_move]['action']
                
            # check repeated move
            self.repeated_move = g.state[self.move_t] != -100

            if self.repeated_move == False:
                # Change state
                g.state[self.move_t] = self.position
                            
                # Check winner
                g.check_winner()
                self.win = g.winner[self.position]
            
                # Check finish
                g.check_finish()
            else:
                g.finish = True

    
    def update_Q(self, g, alpha = 1, gamma = 0.7):
        reward = np.float(self.win)
        if self.repeated_move == True:
            reward = -1
        
        Q_index = self.Q.state.apply(lambda x: np.array_equal(x, self.state_t))   
        Q_A_index = (self.Q.action == self.move_t) * Q_index
        
        if sum(Q_A_index) == 0:
            if reward != 0:
                new_Q = reward
                self.Q = self.Q.append(pd.DataFrame([{'state': self.state_t, 'action': self.move_t, 'Q': new_Q}]))
            else:
                self.Q = self.Q.append(pd.DataFrame([{'state': self.state_t, 'action': self.move_t, 'Q': 0}]))
        elif reward != 1:
            Q = self.Q.loc[Q_A_index, 'Q']
            future_Q = self.Q[self.Q['state'].apply(lambda x: np.array_equal(x, g.state))].Q
            if len(future_Q) > 0:
                new_Q = Q + alpha * (reward + gamma * max(future_Q) - Q)
                self.Q.loc[Q_A_index, 'Q'] = new_Q
   
        

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

for i in range(1000):
    game_1 = game()
    epsilon = 1 - i/1000
    while game_1.finish == False:
        player_1.move(game_1, rand_prob = epsilon)
        player_0.move(game_1, rand_prob = epsilon)
        player_1.update_Q(g = game_1)
        player_0.update_Q(g = game_1)

game_1.state
player_1.Q


# Evaluation
# random move baseline
#player_1 = player(1)
#player_0 = player(0)

win_record = []
repeated = []
for i in range(1000):
    game_1 = game()
    while game_1.finish == False:
        player_1.move(game_1, rand_prob = 0)
        player_0.move(game_1, rand_prob = 1)
    win_record.append(game_1.winner[1])
    repeated.append(player_1.repeated_move)
    
pd.DataFrame(win_record)[0].value_counts()
pd.DataFrame(repeated)[0].value_counts()


