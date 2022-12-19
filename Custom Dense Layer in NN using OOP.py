''' Creating custom Dense Layer in Neural Networks using OOP, by Saber
'''


import tensorflow as tf
import numpy as np


#_____________________________________________________________________________________________
# inherit from this base class
from tensorflow.keras.layers import Layer #Parent class:

class SimpleDense(Layer): #Child class:  ►#inherit from Layer

    def __init__(self, units=32):#self and default amount of attribute if we dont write anything, it will be unit=32!
        '''Initializes the instance attributes'''
        super(SimpleDense, self).__init__()  #cuz it is inheritance
        self.units = units #define attribute

    def build(self, input_shape):
        '''Create the state of the layer (weights/ kernel)'''
        # initialize the weights
        w_init = tf.random_normal_initializer()  #initialize from random Normal Dist.
        self.w = tf.Variable(name="kernel",
            initial_value=w_init(shape=(input_shape[-1], self.units),
                                 dtype='float32'), #define w attribute with random normal Dist.
            trainable=True)

        # initialize the biases
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(name="bias",
            initial_value=b_init(shape=(self.units,), dtype='float32'),
            trainable=True) #defin b, bias, with zero initializer (setting the initial value to 0)

    def call(self, inputs):
        '''Defines the computation from inputs to outputs'''
        return tf.matmul(inputs, self.w) + self.b #tf.matMul() function is used to compute the dot product of two matrices
    
    
    
    

#____________________________________________________________________
# declare a very simple instance of the class
my_dense = SimpleDense(units=1)

# define an input and feed into the layer
x = tf.ones((1, 1)) #1D array  (tf.ones(shape, dtype, name))
#The tf.ones() function is used to create a new tensor where all elements are set to 1
y = my_dense(x)

# parameters of the base Layer class like `variables` can be used
print(my_dense.variables)


#__________________________________________________________________________
# define the dataset
xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

#########################################
'''Building the model with the aforementioned custom dense layer:
'''
# use the Sequential API to build a model with our custom layer
input_layer = tf.keras.Input(shape=(1,)) # Added Newly
my_layer = SimpleDense(units=1)
model = tf.keras.Sequential([input_layer, my_layer]) # Added input_layer here.

# configure and train the model
model.compile(optimizer='sgd', loss='mean_squared_error')
model.fit(xs, ys, epochs=500,verbose=0)


###########################
'''Prediction:
'''
# perform inference
print(model.predict([10.0]))

# see the updated state of the variables
print(my_layer.variables)  ## y= 2x-1


