"""
This script runs CCM on three example datasets. 
These datasets correspond to the synthetic datasets in the paper.

Type of the datasets are binary classification, 
categorical classification, and regression respectively.
"""
from __future__ import print_function
import numpy as np
from make_synthetic_data import generate_data
import sys
sys.path.append('core')
import ccm 

print('Running CCM on the orange skin dataset...')
X, Y = generate_data(n=100, datatype='orange_skin', seed = 0)
epsilon = 0.001; num_features = 4; type_Y = 'binary'
rank = ccm.ccm(X, Y, num_features, type_Y, 
	epsilon, iterations = 100, verbose = False)
selected_feats = np.argsort(rank)[:4]
print('The four features selected by CCM on the orange skin dataset are features {}'.format(selected_feats))

print('-------------------------------------------')
print('Running CCM on the XOR dataset...')
X, Y = generate_data(n=100, datatype='XOR', seed = 0)
epsilon = 0.001; num_features = 3; type_Y = 'categorical'
rank = ccm.ccm(X, Y, num_features, type_Y, epsilon, iterations = 100, verbose = False) 
selected_feats = np.argsort(rank)[:3]
print('The three features selected by CCM on the XOR dataset are features {}'.format(selected_feats))

print('-------------------------------------------')
print('Running CCM on the nonlinear regression dataset...')
X, Y = generate_data(n=100, datatype='regression', seed = 0)
epsilon = 0.1; num_features = 4; type_Y = 'real-valued'
rank = ccm.ccm(X, Y, num_features, type_Y, epsilon, verbose = False) 
selected_feats = np.argsort(rank)[:4]
print('The four features selected by CCM on the nonlinear regression dataset are features {}'.format(selected_feats))

print('-------------------------------------------')
print('Running CCM on the approximate nonlinear regression dataset...')
X, Y = generate_data(n=100, datatype='regression_approx', seed = 0)
epsilon = 0.1; num_features = 4; type_Y = 'real-valued'
rank = ccm.ccm(X, Y, num_features, type_Y, epsilon, D_approx=5, verbose = False) 
selected_feats = np.argsort(rank)[:4]
print('The four features selected by CCM on the approximate nonlinear regression dataset are features {}'.format(selected_feats))
