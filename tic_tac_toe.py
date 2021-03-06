# A Python script to perform deep reinforcement learning on Tic-Tac-Toe
# Q-network
# Asynchronous Methods for Deep Reinforcement Learning
# https://arxiv.org/abs/1602.01783
# Tom Li 
# April 2017

import numpy as np
import timeit
import random
import matplotlib.pyplot as plt        
import pandas as pd


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
def rewards(state, previou_state, action):
    # If move on occupied space, then penalise
    non_empty = [i for i, s in enumerate(previou_state) if s != -1]   
    if(any(i == action for i in non_empty)):
        return -1
    elif game_status(state, 0) == 1:
        return 1
    else:
        return 0

S = np.array([1,1,-1,0,0,-1,-1,-1,-1], dtype = float)    
S2 = np.array([1,1,-1,0,0,0,-1,-1,-1], dtype = float)    
rewards(S2, S, 5)

# Emulate---------------------------------------------------
def Emulate(state, action, turn):
    state_temp = np.array(state)
    state_temp[action] = turn
    return state_temp

Emulate(S, 7, 0)

# Make a move----------------------------------------------
def make_a_move(S, epsilon):
    Q = model.predict(S.reshape(1,9), batch_size=1).tolist()[0]
    if (random.random() < epsilon): #choose random action
        action = np.random.randint(0,9)
    else: #choose best action from Q(s,a) values
        action = (np.argmax(Q))
    return action

# 'Flip' the game board, 0 <-> 1-------------------------------
def flip_board(new_S):
    empty = [i for i, s in enumerate(new_S) if s == -1]
    S = np.array(1 - new_S)
    S[empty] = -1 

    return S 
        
    
# Deep neural net----------------------------------------------
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras import backend as k

model = Sequential()
model.add(Dense(100, init='lecun_uniform', input_shape=(9,)))
model.add(Activation('relu'))
#model.add(Dropout(0.5))

model.add(Dense(100, init='lecun_uniform'))
model.add(Activation('relu'))
#model.add(Dropout(0.5))

model.add(Dense(9, init='lecun_uniform'))
model.add(Activation('linear'))

rms = RMSprop()
model.compile(loss='mse', optimizer=rms)
model.predict(S.reshape(1,9), batch_size=1)

# Learning-------------------------------------------------------
def learning(n_round, WinExp, Exp, epsilon):

    initial_state = np.repeat(-1.0, 9, axis = 0)
    gamma = 0.5
#    epsilon = 0.1
#    Exp = [] # experience
#    WinExp = [] # winning experience
    Exp_size = 500
    WinExp_size = 200
    batch_size = 5
    r_record = []
    for rounds in range(n_round):
        # Assume I play 0, opponent plays 1
        turn = np.random.randint(0,2)
        S = np.array(initial_state)
        if turn == 1:
            action = make_a_move(S, epsilon)
            S = Emulate(S, action, turn)
           
        finished = 0
        counter = 0
        
        while finished != 1:
          
            action = make_a_move(S, epsilon)
            new_S = Emulate(S, action, 0)
            r = rewards(new_S, S, action)
            finished = check_finish(new_S)

            # If not finished, let oppenent plays a move
            if r == 0 and finished != 1:
                new_S = flip_board(new_S)
                S_1 = new_S
                action_1 = make_a_move(S_1, 0)
                new_S = Emulate(new_S, action_1, 0)
                r_1 = rewards(new_S, S_1, action_1)
                finished = check_finish(new_S)
                new_S = flip_board(new_S)
                if r_1 != 0 or finished == 1:
                    break
                
            # memorise experience
            # Combine both winning and losing experience
            if r >= 0:
                if len(WinExp) > WinExp_size:
                    WinExp = WinExp[1:] # remove the first
                WinExp.append([S, new_S, action, r])
            else:
                if len(Exp) > Exp_size:
                    Exp = Exp[1:] # remove the first
                Exp.append([S, new_S, action, r])

            AllExp = [WinExp, Exp]
            AllExp = [AllExp[i][0] for i in range(len(AllExp)) if AllExp[i] != []]
                
            # Learning
            minibatch = random.sample(AllExp, min(batch_size, len(AllExp)))
            X_train = []
            Y_train = []
            for memory in minibatch:
                S_mem, new_S_mem, action_mem, r_mem = memory 
                old_Q = model.predict(S_mem.reshape(1,9), batch_size=1).tolist()[0]
                new_Q = model.predict(new_S_mem.reshape(1,9), batch_size=1).tolist()[0]
                y = np.array(old_Q)
                finished_mem = check_finish(new_S_mem)
                if r_mem == -1:
                    finished_mem == 1
                y[action_mem] = r_mem + (1 - finished_mem) * gamma * max(new_Q)
                
                X_train.append(S_mem)
                Y_train.append(y)
                  
            model.fit(np.array(X_train), np.array(Y_train), batch_size=1, nb_epoch=1, verbose=0)
            
            ### Learning from opponent's move (learning defensive move)
    #        if game_status(new_S, 0) == 1 and counter != 0:
    #            Q = model.predict(last_S.reshape(1,9), batch_size=1).tolist()[0]
    #            y = np.array(Q)
    #            y[last_action] = -1
    #            model.fit(last_S.reshape(1,9), y.reshape(1,9), batch_size=1, nb_epoch=1, verbose=0)
    # 
    #        last_S = np.array(S)
    #        last_action = action
    
            finished = check_finish(new_S)
            if r != 0:
                finished = 1
            
            S = new_S

            counter = counter + 1
            r_record.append(r)
    return r_record
    #        print S.reshape(3,3)
#        print rounds

learning(100)

# Play with me
initial_state = np.repeat(-1.0, 9, axis = 0)
turn = 1
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


S = np.array([0,-1,0,0,1,-1,1,1,-1], dtype = float)    
S.reshape(3,3)
np.argmax(model.predict(S.reshape(1,9), batch_size=1).tolist()[0])


# Test play
def test_play():
    win = 0
    for game in range(1000):
        initial_state = np.repeat(-1.0, 9, axis = 0)
#        turn = np.random.randint(0, 2)
        turn = 0
        S = np.array(initial_state)
        finished = 0
        penalty = 0
        while finished != 1:
            non_empty = []
            if turn == 1:
                empty = [i for i, s in enumerate(S) if s == -1]
                var =  random.sample(empty, 1)
                S[var] = 1
            else:
                non_empty = [i for i, s in enumerate(S) if s != -1]   
                Q = model.predict(S.reshape(1,9), batch_size=1).tolist()[0]
                action = (np.argmax(Q))
                S[action] = 0

            finished = check_finish(S)
            
            if(any(i == action for i in non_empty)):
                penalty = 1
                finished = 1
            turn = 1 - turn
        if game_status(S, 0) == 1 and penalty != 1:
            win += 1
        
    return win

    
result = []
Exp = [] # experience
WinExp = [] # winning experience
epsilon = 0.5

for i in range(500):
    epsilon = epsilon*0.95
    r_record = learning(1000, WinExp, Exp, epsilon)
    result.append(test_play())
    print pd.Series(r_record).value_counts()
    print np.mean(r_record)
    plt.figure()
    plt.plot(range(len(result)), result)
    plt.show()

        