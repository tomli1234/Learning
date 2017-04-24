# A Python script to perform deep reinforcement learning on Tic-Tac-Toe
# Asynchronous Methods for Deep Reinforcement Learning
# https://arxiv.org/abs/1602.01783
# Tom Li 
# April 2017

import numpy as np
import timeit
import time

# Identify possible moves-----------------------------------
S = np.repeat(-1, 9, axis = 0)
def possible_moves(state, turn):
    move = [i for i, s in enumerate(S) if s == -1]
    S_next = []
    for i in move:
        s_next = state.tolist()
        s_next[i] = turn
        S_next.append(s_next)
    return S_next
            
possible_moves(S, 0)
timeit.timeit(lambda: possible_moves(S, 0), number = 1)

