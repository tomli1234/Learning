# A Python script to perform deep reinforcement learning on Tic-Tac-Toe
# Asynchronous Methods for Deep Reinforcement Learning
# https://arxiv.org/abs/1602.01783
# Tom Li 
# April 2017

import numpy as np
import timeit

# Identify possible moves-----------------------------------
def possible_moves(state, turn):
    empty = [i for i, s in enumerate(S) if s == -1]
    S_next = []
    for i in empty:
        s_next = state.tolist()
        s_next[i] = turn
        S_next.append(s_next)
    return S_next

S = np.repeat(-1.0, 9, axis = 0)
possible_moves(S, 0)
timeit.timeit(lambda: possible_moves(S, 0), number = 1000)/1000

# Check game status---------------------------------------
def game_status(state, turn):
    empty = [i for i, s in enumerate(S) if s == -1]
    S[empty] = None
    S_mat = S.reshape(3,3)
    colsum = S_mat.sum(axis = 0)
    rowsum = S_mat.sum(axis = 1)
    diagsum_1 = sum(np.diag(S_mat))
    diagsum_2 = sum(np.diag(np.fliplr(S_mat)))
    allsum = np.append(diagsum_2, np.append(diagsum_1, np.append(rowsum, colsum)))
    
    if any(i == 3 for i in allsum):
        return(turn)
    elif any(i == 0 for i in allsum):
        return(1- turn)
    elif any(np.isnan(i) for i in allsum):
        return(1 - turn)
        
S = np.array([1,1,1,0,0,1,-1,-1,-1], dtype = float)    
timeit.timeit(lambda: game_status(S, 0), number = 1000)/1000


  
        
        