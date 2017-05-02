#Define model
from keras.layers import Input, Dense
from keras.models import Model

input = Input(shape=[2])
probs = Dense(1, activation='sigmoid')(input)

model = Model(input=input, output=probs)
model.compile(optimizer='sgd', loss='binary_crossentropy')

#----------------

#-----------------



#Get gradient tensors
weights = model.trainable_weights # weight tensors
weights = [weight for weight in weights if model.get_layer(weight.name[:-2]).trainable] # filter down weights tensors to only ones which are trainable
gradients = model.optimizer.get_gradients(model.total_loss, weights) # gradient tensors

print weights
# ==> [dense_1_W, dense_1_b]

#Define keras function to return gradients
import keras.backend as K

input_tensors = [model.inputs[0], # input data
                 model.sample_weights[0], # how much to weight each sample by
                 model.targets[0], # labels
                 K.learning_phase(), # train or test mode
]

get_gradients = K.function(inputs=input_tensors, outputs=gradients)


#Get gradients of weights for particular (X, sample_weight, y, learning_mode) tuple
from keras.utils.np_utils import to_categorical

inputs = [[[1, 2]], # X
          [1], # sample weights
          [[1]], # y
          0 # learning phase in TEST mode
]

print zip(weights, get_gradients(inputs))
