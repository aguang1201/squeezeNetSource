from tflearn.layers.core import dropout
from tflearn.layers.conv import conv_2d, max_pool_2d,global_avg_pool
from tflearn.layers.estimator import regression
import util

def create_squeezeNet_v1_1(input,num_classes):
    network = conv_2d(input, 64, 3, strides=2, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    # Fire 2
    network = util.create_fire(network,16)
    # Fire 3
    network = util.create_fire(network, 16)
    # MaxPool 3
    network = max_pool_2d(network, 3, strides=2)
    # Fire 4
    network = util.create_fire(network, 32)
    # Fire 5
    network = util.create_fire(network, 32)
    # MaxPool 5
    network = max_pool_2d(network, 3, strides=2)
    # Fire 6
    network = util.create_fire(network, 48)
    # Fire 7
    network = util.create_fire(network, 48)
    # Fire 8
    network = util.create_fire(network, 64)
    # Fire 9
    network = util.create_fire(network, 64)
    # Dropout
    network = dropout(network, 0.5)
    # Conv10
    network = conv_2d(network, num_classes, 1, activation='relu')
    # AVG 1
    network = global_avg_pool(network)

    network = regression(network, optimizer='adam',loss='softmax_categorical_crossentropy',learning_rate=0.0001)

    return network