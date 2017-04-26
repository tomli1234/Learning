# A Python script to perform deep reinforcement learning on Tic-Tac-Toe
# Q-network
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
    state_temp = np.array(state)
    empty = [i for i, s in enumerate(state_temp) if s == -1]
    state_temp[empty] = None
    S_mat = state_temp.reshape(3,3)
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

# Get rewards----------------------------------------------
def rewards(state, previous_action, turn):
    # If move on occupied space, then heavily penalise
    non_empty = [i for i, s in enumerate(state) if s != -1]   
    if(any(i == previous_action for i in non_empty)):
        return -100
    else:
        return game_status(state, turn)

rewards(S, 8, 1)

# Emulate---------------------------------------------------
def Emulate(state, action, turn):
    state_temp = np.array(state)
    state_temp[action] = turn
    return state_temp

Emulate(S, 7, 0)

# Deep neural net----------------------------------------------
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop

model = Sequential()
model.add(Dense(300, init='lecun_uniform', input_shape=(9,)))
model.add(Activation('relu'))
#model.add(Dropout(0.2)) I'm not using dropout, but maybe you wanna give it a try?

model.add(Dense(30, init='lecun_uniform'))
model.add(Activation('relu'))
#model.add(Dropout(0.2))

model.add(Dense(9, init='lecun_uniform'))
model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

rms = RMSprop()
model.compile(loss='mse', optimizer=rms)

model.predict(S.reshape(1,9), batch_size=1)

# Learning-------------------------------------------------------






        
        