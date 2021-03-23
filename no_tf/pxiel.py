import math as m
import os.path
from os import path
import numpy as np
import pickle as jk
import bz2
import random

class Pxiel():
    def __init__(self, X = [], y = [], checkpoint = 'model.h5'):
        # Input data
        self.X = np.array(X)
        # Output data
        self.y = np.array(y)
        # Load model custom or default model.h5
        self.checkpoint = checkpoint
        # Default model is none
        self.model = None
        # Weights, Elastic [1.189 => 1.184...etc]
        self.weights = [0.002, 0.003, 0.004]
        # Here is all weights.
        self.neural_network = []
        # Validation in main.
        # Check if len of x and y eaqu
        if len(self.X) > 0 and len(self.y) > 0:
            if len(self.X.shape) > 0:
                if len(self.y.shape) == 1:
                    if self.X.shape[0] == len(self.y):
                        if self.DotWeight():
                            self.Bias()
                            self.Train()
                    else:
                        print('Shape of X {}, y {} is not equivalent.'.format(self.X.shape, self.y.shape))
                else:
                    print('y {} array must be one dimensional.'.format(self.y.shape))
            else:
                print('X {} array must be two dimensional.'.format(self.X.shape))
        else:
            # Load model
            self.Model()
    # Train
    def Train(self):
        # Validation -> mediana, loss - correct , + wrong
        validation, loss, verbose, samples_r, samples_l = 0, 1, 0, 0, 0
        index = 0
        # Each for X
        for inp in self.X:
            # Check if static weight result equal validation weight result.
            if self.neural_network[index][1] == self.Validation(inp)[1]:
                validation +=1
                loss -= 0.1
                samples_r +=1
            else:
                loss += 0.1
                samples_l +=1
            # Printer result after 100 row.
            if verbose == 100:
                verbose = 0
                print('[Pxiel] - Validation / Loss: {} / Samples R: {} / Samples L: {}'.format(round(loss, 2), samples_r, samples_l))
            verbose +=1
            index +=1
        print('[Pxiel] Accuracy: {}'.format(100.0 *  validation / len(self.neural_network)))
        self.Save()
    
    # Validate input after each loop
    # -0 to 0+
    def Validation(self, ax):
        # Sigmoid function for each value in input list.
        # Get weight for inp [input]
        _near_dot_weight = self.weight(self.sigmoid_as_list(ax)) + random.choice(self.weights)
        # List: Check for instant weight.
        mm = self.ExtractVector(case = self.neural_network)
        # Neural network: weight to list.
        n_dot = np.asarray(self.ExtractVector(case = self.neural_network))
        # Find similary weight of input in neaural network.
        idx = (np.abs(n_dot - _near_dot_weight)).argmin()
        # Check for instant weight.
        mm = self.ExtractVector(case = self.neural_network)
        # Let's try.
        try :
            # Direct access
            return self.neural_network[mm.index(_near_dot_weight)]
        except:
            try:
                # Similar access
                return self.neural_network[idx]
            except:
                # Totaly predicted.
                return self.neural_network[round((idx/2))]

    # Extract value from the 2D array.
    # Return list.
    def ExtractVector(self, case = None):
        return [x[0] for x in case]
    # Weight formula.
    def weight(self, v):
        return sum(v) / len(v) / (sum(v[:round(len(v) / 2)]))
    # Sigmoid functin
    def sigmoid(self, x):
        return (1 / (1 + m.exp(-x)))
    def sigmoid_as_list(self, _list):
        return [self.sigmoid(x) for x in _list]
    # Make weight.
    def DotWeight(self):
        fine_tune , index = [], 0
        # Each value from x
        for k in self.X:
            v = []
            # Get sigmoid function for each value
            # Get weight of input case: X
            weight = self.weight(self.sigmoid_as_list(k))
            # Fine tune same weight
            if weight in fine_tune:
                weight = weight + random.choice(self.weights)
            fine_tune.append(weight)
            # Append weight to NN and result.
            self.neural_network.append([weight, self.y[index]])
            index +=1
        return True

    def Bias(self):
        # Check the bias tune.
        bias, index = [], 0
        # Each weight
        for weight in self.neural_network:
            # find all same weight
            for each_weight in self.neural_network:
                # Weight == ?
                if each_weight[0] == weight[0]:
                    # If weight recorded ?
                    if weight[0] not in bias:
                        # Assign bias 
                        bias.append([weight[0], 0.0001])
                    # Deep start point is zero.
                    deep = 0
                    for b in bias:
                        # Check index and bias ==
                        if b[0] == weight[0]:
                            # Add bias to weight.
                            deep += 0.0001
                            bias[index][1] = deep
                    # Update NN
                    self.neural_network[index][0] = weight[0] - deep
            index  +=1

    def Predict(self, input):
        # Sigmoid list : input
        ax = self.sigmoid_as_list(input)
        # Get weight from input
        weight = self.weight(ax)
        # Extract all weights: vector
        vector = np.asarray(self.ExtractVector(case = self.model))
        # Get similray weight [clossest]
        point = (np.abs(vector - weight)).argmin() 
        # List: Check for instant weight.
        mm = self.ExtractVector(case = self.model)
        # Let's try.
        try :
            # Direct access
            return self.model[mm.index(weight)][1]
        except:
            try:
                # Similar access
                return self.model[point][1]
            except:
                # Totaly predicted.
                return self.model[round((point/2))][1]

    # Load model
    def Model(self):
        # Check if model exists.
        if path.exists(self.checkpoint):
            # Load model like R+BINARY readable.
            _m = bz2.BZ2File(self.checkpoint, 'rb')
            # Define self.model : like continue....
            self.model = jk.load(_m)
        else:
            # Model does not exists return false.
            print('Model "{}" does not exists.'.format(self.checkpoint))
    
    # Save the model.
    def Save(self):
        # Decode model via bz2, binary.
        _m = bz2.BZ2File(self.checkpoint, 'w')
        # Exoirt model, ! important protocol=2 [encode binary]
        jk.dump(self.neural_network, _m, protocol=2)
