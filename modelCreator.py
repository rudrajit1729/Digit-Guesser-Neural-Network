# Used to train and save a neural network on the mnist dataset

# Importing the libraries
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Part 1 - Data Preprocessing

# Importing the dataset
'''
    MNIST (Modified National Institute of Standards and Technology database)
    This is a dataset of 60,000 28x28 grayscale images of the 10 digits, 
    along with a test set of 10,000 images.
    More info can be found at the (MNIST homepage)[yann.lecun.com/exdb/mnist/].
 '''
mnist = tf.keras.datasets.mnist

# Loading and splitting data into training and test sets

(x_train, y_train),(x_test, y_test) = mnist.load_data()
'''
    Returns:
    Tuple of Numpy arrays: (x_train, y_train), (x_test, y_test).
    x_train, x_test: uint8 arrays of grayscale image data with shapes (num_samples, 28, 28).
    y_train, y_test: uint8 arrays of digit labels (integers in range 0-9) with shapes (num_samples,).
'''

# Normalizing the feature matrix
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Assigning 1 to all non zero pixel values
for train in range(len(x_train)):
    for row in range(28):
        for x in range(28):
            if x_train[train][row][x] != 0:
                x_train[train][row][x] = 1
                
# Part 2 - Creating the Neural Network
                
try:
    # Try loading the model
    model = tf.keras.models.load_model('m.model')
    print("Model loaded successfully...")
except:    
    # Model creation
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10)
    model.save('m.model')
    print("Model saved successfully...")

# Evaluate the model
scores = model.evaluate(x_test,y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

