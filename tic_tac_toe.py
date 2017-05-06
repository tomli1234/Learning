# A Python script to perform deep reinforcement learning on Tic-Tac-Toe
# Q-network
# Asynchronous Methods for Deep Reinforcement Learning
# https://arxiv.org/abs/1602.01783
# Tom Li 
# April 2017

import numpy as np
import timeit
import random

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
    else:
        return 0
        
S = np.array([0,0,0,-1,-1,-1,-1,-1,-1], dtype = float)    
timeit.timeit(lambda: game_status(S, 1), number = 1000)/1000

def check_finish(state):
    empty = [i for i, s in enumerate(state) if s == -1]
    if len(empty) == 0:
        return 1
    elif (game_status(state, 0) == 1) or (game_status(state, 1) == 1):
        return 1
    else:
        return 0

S = np.array([0,-1,1,1,0,-1,-1,-1,0], dtype = float)    
check_finish(S)
        
        
# Get rewards----------------------------------------------
def rewards(state, previou_state, action, turn):
    # If move on occupied space, then penalise
    non_empty = [i for i, s in enumerate(previou_state) if s != -1]   
    if(any(i == action for i in non_empty)):
        return -1
    else:
        return game_status(state, turn)

S = np.array([1,1,1,0,0,-1,-1,1,1], dtype = float)    
S2 = np.array([1,1,1,0,0,-1,-1,1,0], dtype = float)    
rewards(S2, S, 8, 1)

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
from keras import backend as k

model = Sequential()
model.add(Dense(50, init='lecun_uniform', input_shape=(9,)))
model.add(Activation('relu'))
#model.add(Dropout(0.5))

model.add(Dense(50, init='lecun_uniform'))
model.add(Activation('relu'))
#model.add(Dropout(0.5))

model.add(Dense(9, init='lecun_uniform'))
model.add(Activation('linear'))

rms = RMSprop()
model.compile(loss='mse', optimizer=rms)

model.predict(S.reshape(1,9), batch_size=1)

# Learning-------------------------------------------------------
initial_state = np.repeat(-1.0, 9, axis = 0)
gamma = 0.5
epsilon = 0.1
D = [] # experience

for rounds in range(5000):
    # Assume I play 0, opponent plays 1
    turn = 0
    S = np.array(initial_state)
    finished = 0
    counter = 0
    
    while finished != 1:
      
        Q = model.predict(S.reshape(1,9), batch_size=1).tolist()[0]
        if (random.random() < epsilon): #choose random action
            action = np.random.randint(0,8)
        else: #choose best action from Q(s,a) values
            action = (np.argmax(Q))
        new_S = Emulate(S, action, turn)
        r = rewards(new_S, S, action, turn)
        
        # memorise experience
        D.append([S, new_S, action, r])
        memory = np.random.randint(0, len(D))
        S, new_S, action, r = D[memory]
        
        new_Q = model.predict(new_S.reshape(1,9), batch_size=1).tolist()[0]
        y = np.array(Q)
        finished = check_finish(new_S)
        if r == -1:
            finished == 1
        y[action] = r + (1 - finished) * gamma * max(new_Q)
        model.fit(S.reshape(1,9), y.reshape(1,9), batch_size=1, nb_epoch=1, verbose=0)
        
        ### Learning from opponent's move (learning defensive move)
#        if game_status(new_S, 0) == 1 and counter != 0:
#            Q = model.predict(last_S.reshape(1,9), batch_size=1).tolist()[0]
#            y = np.array(Q)
#            y[last_action] = -1
#            model.fit(last_S.reshape(1,9), y.reshape(1,9), batch_size=1, nb_epoch=1, verbose=0)
# 
#        last_S = np.array(S)
#        last_action = action
            
        # 'Flip' the game board, 0 <-> 1
        empty = [i for i, s in enumerate(new_S) if s == -1]
        S = np.array(1 - new_S)
        S[empty] = -1

        counter = counter + 1
#        print S.reshape(3,3)
    print rounds


# Play with me
initial_state = np.repeat(-1.0, 9, axis = 0)
turn = 0
S = np.array(initial_state)
finished = 0
while finished != 1:
    if turn == 1:
        print S.reshape(3,3)
        var = raw_input("Please enter something: ")
        S[var] = 1
    else:
        Q = model.predict(S.reshape(1,9), batch_size=1).tolist()[0]
        action = (np.argmax(Q))
        S[action] = 0
    finished = check_finish(S)
    turn = 1 - turn
print S.reshape(3,3)


S = np.array([1,1,-1,-1,-1,-1,0,0,-1], dtype = float)    
S.reshape(3,3)
np.argmax(model.predict(S.reshape(1,9), batch_size=1).tolist()[0])























        
        