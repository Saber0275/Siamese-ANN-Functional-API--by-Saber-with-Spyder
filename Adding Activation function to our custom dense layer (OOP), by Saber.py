'''Adding Activation function to our custom dense layer (OOP), by Saber
'''

import tensorflow as tf
from tensorflow.keras.layers import Layer #for creating our custom dense layer

#_______________________________________________________________________________

class SimpleDense(Layer): # Parent class: Layer, Child class: SimpleDense

    # add an activation parameter
    def __init__(self, units=32, activation=None):  #the default of activation is  None
        super(SimpleDense, self).__init__() #cuz inheritance
        self.units = units #attribute1
        
        # define the activation to get from the built-in activation layers in Keras
        self.activation = tf.keras.activations.get(activation) #attribute2


    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(name="kernel",
            initial_value=w_init(shape=(input_shape[-1], self.units),
                                 dtype='float32'),
            trainable=True) #w/kernel is trainable (we can do gradient/derivative)
        b_init = tf.zeros_initializer() #set the initial value of bias to 0
        self.b = tf.Variable(name="bias",
            initial_value=b_init(shape=(self.units,), dtype='float32'),
            trainable=True) #bias is trainable 
        super().build(input_shape)


    def call(self, inputs):
        
        # pass the computation to the activation layer
        return self.activation(tf.matmul(inputs, self.w) + self.b)
#________________________________________________________________________________
       
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    SimpleDense(128, activation='relu'),  #adding activation function to our layer
    tf.keras.layers.Dropout(0.2), #to challenge the learning (dropping some neurons in calculation)
    tf.keras.layers.Dense(10, activation='softmax') #softmax► a generalized logistic reg for classification (units=10 classes)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        