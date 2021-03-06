#!/usr/bin/python

pip install matplotlib-venn
pip install numpy
import numpy as np
import pandas as pd


features = np.array([
    [-0.6508, 0.1097, 4.0009],
    [-1.4492, 0.8896, 4.4005],
    [2.0850, 0.6876, 12.0710],
    [0.2626, 1.1476, 7.7985],
    [0.6418, 1.0234, 7.0427],
    [0.2569, 0.6730, 8.3265],
    [1.1155, 0.6043, 7.4446],
    [0.0914, 0.3399, 7.0677],
    [0.0121, 0.5256, 46316],
    [-0.0429, 0.4660, 5.4323]
])

instance = features[1]
x0 = instance[0]
#print (x0, instance)

labels = np.array([-1,-1,-1,1,1,-1,1,-1,1,1])

w = [0.1, 0.1, 0.1] ## initial weights
print (w[1])

w0 = 0.1 ## bias

threshold = 0.5

learning_rate = 0.1
epoch = 30

for j in range(0, epoch):
    print ("epoch: ", j)
    global_delta = 0
    
    for i in range (0, features.shape[0]):

        actual = labels[i]
        instance = features[i]

        x1 = instance[0]
        x2 = instance[2]
        x3 = instance[1]
        summation = x1*w[0] + x2*w[1] + x3*w[2] + w0 ##summation
        
        ## activation function
        if summation >= threshold:
            trigger = 1
        else:
            trigger = -1

        delta = actual - trigger
        global_delta = global_delta + abs(delta)

        print("perceptron: ", trigger, "actual: ", actual, "summation", summation, "(error: ", delta, ")")

        w[0] = w[0] + delta * learning_rate
        w[1] = w[1] + delta * learning_rate
        w[2] = w[2] + delta * learning_rate

    print("-----------------------------")

    if global_delta == 0:
        break


   
