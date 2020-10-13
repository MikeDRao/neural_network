#Michael Rao
#1001558150
#[[random.randrange(-0.5,0.5)]*D]*L    weighted matrix

import os
import sys
from math import pi
from math import exp
import array
import random
import numpy as np


# Load file
def load_file(file_name):
    dataset = list()
    with open(file_name, 'r') as file:
        lines = file.readlines()
        for line in lines:
            dataset.append(line.split())
    return dataset

# Convert string column to float
def create_floats(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

# Convert string to int
def create_int(dataset, column):
    class_val = [row[column] for row in dataset]
    unique = set(class_val)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = int(value)
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

# Find the min/max value
def abs_max(dataset):
    max_ret = 0
    minmax = list()
    for row in dataset:
        val = 0
        for i in range(len(row) - 1 ):
            val = abs(row[i])
            if val > max_ret:
                max_ret = val
    return max_ret

# Divide all values by abs max
def normalize_attributes(dataset, max_value):
    for row in dataset:
        for i in range(len(row) - 1):
            row[i] = row[i]/max_value

# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation

# Transfer neuron activation
def transfer(activation):
    return 1.0/(1.0 + exp(-activation))

# Forward propagate input
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


# Calculate the derivative of a neuron output
def transfer_derivative(output):
    return output * (1.0 - output)

# Initialze input layer
def initialze_network(num_layer, unit_per_layer,n_outputs,n_inputs):
    network = list()
    if num_layer > 2:
        for i in range(num_layer - 2):
            hidden_layer = [{'weights':[random.uniform(-0.5,0.5) for j in range(n_inputs +1)]}for j in range(unit_per_layer)]
            network.append(hidden_layer)    
        output_layer = [{'weights':[random.uniform(-0.5,0.5) for j in range(unit_per_layer + 1)]}for j in range(n_outputs)]
        network.append(output_layer)
        return network
    else:
        output_layer = [{'weights':[random.uniform(-0.5,0.5) for j in range(n_inputs + 1)]}for j in range(n_outputs)]
        network.append(output_layer)
    return network

# Backpropagate
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# Update network weights with error
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1]+= l_rate * neuron['delta']

# Train a network for a fixed number epochs
def train_network(network, training_file, l_rate, epochs, n_outputs):
    for epoch in range(epochs):
        for row in training_file:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        l_rate = (l_rate*0.98)                

# Make a prediction
def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))

def neural_network():
    if(len(sys.argv) < 6):
        print("Insufficient command line args")
        exit()

    training_file = load_file(sys.argv[1])
    test_file = load_file(sys.argv[2])
    num_layer = int(sys.argv[3])
    unit_per_layer = int(sys.argv[4])
    rounds = int(sys.argv[5])

    for i in range(len(training_file[0])):
        create_floats(training_file,i)
    lookup = create_int(training_file, len(training_file[0]) - 1)
    absolute_max = abs_max(training_file)
    normalize_attributes(training_file,absolute_max)

    for i in range(len(test_file[0])):
        create_floats(test_file,i)
    absolute_max = abs_max(test_file)
    normalize_attributes(test_file,absolute_max)

    n_outputs = lookup[max(lookup)] + 1
    n_inputs = len(training_file[0]) - 1

    network = initialze_network(num_layer,unit_per_layer,n_outputs,n_inputs)

    train_network(network, training_file, 1, rounds, n_outputs)
    predictions = list()
    tot_acc = 0
    for row in range(len(test_file)):
        acc = 0
        prediction = predict(network,test_file[row])
        predictions.append(prediction)
        if(prediction == test_file[row][-1]):
            acc = 1
            tot_acc += 1
        print("ID={0:5d}, predicted={1:3d}, true =  {2:3d}, accuracy=".format(row + 1, prediction,int(test_file[row][-1])), acc)
    print("classification accuracy = {0:6.4f}".format(tot_acc/(row+1)))
if __name__ == '__main__':
    neural_network()