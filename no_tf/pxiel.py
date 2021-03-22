import math as m
import os.path
from os import path
import numpy as np
import pickle as jk
import bz2

class Pxiel():
    def __init__(self, X = [], y = [], l_r = 0, elastic = False, checkpoint = 'model.h5'):
        # Input data
        self.X = np.array(X)
        # Output data
        self.y = np.array(y)
        # Learning rate
        self.l_r = l_r
        # Retry per step if validation false.
        self.elastic = elastic
        # Load model custom or default model.h5
        self.checkpoint = checkpoint
        # Default model is none
        self.model = None

        self.Nodes = 0

        # Storage is a RAM.
        self.neural_network = []

        # Validation in main.
        if len(self.X) > 0 and len(self.y) > 0:
            # Start train
            if len(self.X.shape) == 2:
                if len(self.y.shape) == 1:
                    if self.X.shape[0] == len(self.y):
                        if self.l_r <= 0.1:
                            if self.elastic == True or self.elastic == False:
                                if self.DotVector():
                                    self.Train()
                            else:
                               print('Retry per failed validation must be betwen 0 and 100, default is 0')
                        else:
                            print('Learning rate must be betwen 0 and 1, default is 1')
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
        model, dot, validation, loss, verbose = None, self.neural_network, 0, 1, 0
        samples_r, samples_l = 0, 0
        index = 0
        for input in self.X:
            if self.y[index] == self.Validation(input)[1]:
                validation +=1
                loss -= 0.1
                samples_r +=1
            else:
                loss += 0.1
                samples_l +=1

            index +=1
            if verbose == 100:
                verbose = 0
                print('[Pxiel] - Validation / Loss: {} / Samples R: {} / Samples L: {}'.format(round(loss, 2), samples_r, samples_l))
            verbose +=1
        print('[Pxiel] Accuracy: {}'.format(100.0 *  validation / len(self.neural_network)))
        self.Save()
    
    # Validate input after each loop
    # -0 to 0+
    def Validation(self, ax, elastic = False):
        result , input, dot = [], [], self.neural_network

        [(lambda v: input.append((1 / (1 + m.exp(-v)))))(z) for z in ax]

        #for vector in dot:
        #Do vector
        if elastic == False:
            _near_dot_weight = sum(input) / len(input)
        else :
            _near_dot_weight = input
        n_dot = np.asarray(self.ExtractVector(case = self.neural_network))
        idx = (np.abs(n_dot - _near_dot_weight)).argmin()

        try:
            return self.neural_network[idx]
        except:
            return self.neural_network[round((idx/2)-1)]

    def ExtractVector(self, case = None):
        return [x for x in case]

    # Transform data to dot then convert to vector, tensor dot.
    # Try with pi*
    # Sigmoid
    def DotVector(self):
        index, dot = 0, []
        for k in self.X:
            v = []
            for z in k:
                v.append(1 / (1 + m.exp(-z)))
            vector = sum(v) / len(v)
            self.neural_network.append([vector, self.y[index]])
            index +=1
        return True

    def Predict(self, input):
        result , ax = [], []
        [(lambda v: ax.append((1 / (1 + m.exp(-v)))))(z) for z in input]
        weight = sum(ax) / len(ax)
        vector = np.asarray(self.ExtractVector(case = self.model))
        point = (np.abs(vector - weight)).argmin() 
        return self.model[point][1]

    # Load model
    def Model(self):
        if path.exists(self.checkpoint):
            _m = bz2.BZ2File(self.checkpoint, 'rb')
            self.model = jk.load(_m)
        else:
            print('Model "{}" does not exists.'.format(self.checkpoint))
    
    # Save the model.
    def Save(self):
        _m = bz2.BZ2File(self.checkpoint, 'w')
        jk.dump(self.neural_network, _m, protocol=2)
