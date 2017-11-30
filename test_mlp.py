import numpy as np
from pprint import pprint
from numpy import linalg as LA
import time
import gzip,pickle

f=gzip.open('../mnist.pkl.gz')
data=pickle.load(f)
# data[0][0]: matrice de train data
# data[0][1]: vecteur des train labels

# data[1][0]: matrice de valid data
# data[1][1]: vecteur des valid labels

# data[2][0]: matrice de test data
# data[2][0]: vecteur des test labels

traindata_mnist = np.concatenate((data[0][0], np.array([data[0][1]]).T), axis=1)
validdata_mnist = np.concatenate((data[1][0], np.array([data[1][1]]).T), axis=1)
testdata_mnist = np.concatenate((data[2][0], np.array([data[2][1]]).T), axis=1)

mlp = mlp_mini_batch_matrix(0.09, 784, 500, 10, epsilon=0.0001, minibatch_size=100, 
                              num_iterations=100,l11=0.00005, l12=0.00015, l21=0.00005, l22=0.00015)
mlp.train(traindata_mnist, validdata_mnist, testdata_mnist)
