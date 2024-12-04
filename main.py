import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras

def load_idx1_to_np(path_to_file):

    #open file
    f = open(path_to_file, "rb")

    #Get data based on descriptions 
    magic_number = int.from_bytes(f.read(4), 'big')
    number_of_items = int.from_bytes(f.read(4), 'big')
    
    #read the rest data as unsigned integers
    stuff = np.frombuffer(f.read(), np.uint8).reshape(number_of_items)

    #change the label to the format of this NN's output
    output = np.zeros((number_of_items, 10))
    for i in range(0, number_of_items, 1):
        output[i,stuff[i]] = 1

    output = output.astype(int)

    f.close()
        
    return output

def load_idx3_to_np(path_to_file):

    #open file
    f = open(path_to_file, "rb")

    #Get data based on descriptions 
    magic_number = int.from_bytes(f.read(4), 'big')
    number_of_images = int.from_bytes(f.read(4), 'big')
    number_of_rows = int.from_bytes(f.read(4), 'big')
    number_of_columns = int.from_bytes(f.read(4), 'big')
    
    #read the rest data as unsigned integers
    #normalize input and add the last parameter 1 for the channel for conv
    output = np.frombuffer(f.read(), np.uint8).reshape((number_of_images, number_of_rows, number_of_columns, 1)) / 255

    f.close()
        
    return output

#get datas
training_datas = load_idx3_to_np('train-images.idx3-ubyte')
training_classes = load_idx1_to_np('train-labels.idx1-ubyte')

testing_datas = load_idx3_to_np('t10k-images.idx3-ubyte')
testing_classes = load_idx1_to_np('t10k-labels.idx1-ubyte')

#model set up-------------------------------------------------------------------------------------------------------------

#batch size and epoch
batchsize = 32
e = 10

#exactly as stated in document: conv -> pooling -> conv -> pooling -> 1d flatten -> 100 neurons -> 100 neurons -> 10 output
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='valid'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='valid'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=100, activation='sigmoid'),
    tf.keras.layers.Dense(units=100, activation='sigmoid'),
    tf.keras.layers.Dense(units=10, activation='sigmoid')
])

#exactly as stated in document: SGD, learning rate 10, MSE, and accuracy
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=10.0), loss='mean_squared_error', metrics=['accuracy'])
model.fit(training_datas, training_classes, epochs=e, batch_size=batchsize)

#testing-------------------------------------------------------------------------------------------------------------

print("testing time!")

loss, accuracy = model.evaluate(testing_datas, testing_classes)
print(accuracy)