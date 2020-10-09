#Michael Rao
#1001558150

import os
import sys
from math import pi
from math import ex
import array
import random

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

# Initialze input layer
def initialze_input_layer(L,D):
    Z = [[random.randrange(-0.5,0.5)]*D]*L
    return Z

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
    print(training_file)

if __name__ == '__main__':
    neural_network()