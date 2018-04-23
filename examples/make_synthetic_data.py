"""
This script contains functions for generating synthetic data. 
""" 
from __future__ import print_function

import numpy as np  

def generate_data(n=100, datatype='', seed = 0):
    """
    Generate data (X,y)

    Args:
        n(int): number of samples 

        datatype(string): The type of data 
        choices: 'orange_skin', 'XOR', 'regression'.

        seed: random seed used

    Return: 
        X(float): [n,d].  

        y(float): n dimensional array. 
    """
    np.random.seed(seed)


    if datatype == 'orange_skin': 
        X = []

        i = 0 
        while i < n//2:
            x = np.random.randn(10) 
            if 9 < sum(x[:4]**2) < 16:
                X.append(x)
                i += 1
        X = np.array(X)

        X = np.concatenate((X, np.random.randn(n//2, 10)))

        y = np.concatenate((-np.ones(n//2), np.ones(n//2)))

        perm_inds = np.random.permutation(n)
        X, y = X[perm_inds], y[perm_inds]

    elif datatype == 'XOR':  
        X = np.random.randn(n, 10)
        y = np.zeros(n) 
        splits = np.linspace(0,n,num = 8+1,dtype = int) 
        signals = [[1,1,1],[-1,-1,-1],[1,1,-1],[-1,-1,1],[1,-1,-1],[-1,1,1],[-1,1,-1],[1,-1,1]]
        for i in range(8):
            X[splits[i]:splits[i+1],:3] += np.array([signals[i]]) 
            y[splits[i]:splits[i+1]] = i // 2

        perm_inds = np.random.permutation(n)
        X, y = X[perm_inds], y[perm_inds]

    elif datatype == 'regression': 
        X = np.random.randn(n, 10) 

        y = -2 * np.sin(2*X[:,0]) + np.maximum(X[:,1], 0) + X[:,2] + np.exp(-X[:,3]) + np.random.randn(n) 

    elif datatype == 'regression_approx': 
        X = np.random.randn(n, 10) 

        y = -2 * np.sin(2*X[:,0]) + np.maximum(X[:,1], 0) + X[:,2] + np.exp(-X[:,3]) + np.random.randn(n) 

    return (X, y) 
