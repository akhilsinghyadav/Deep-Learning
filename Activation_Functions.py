# some useful Activation function used in Deep Learning Algorithms

# function 1 - Logistic Sigmoid

def logistic_sigmoid(x):
    sigma = 1/(1 + exp(-x))
    return sigma
#range - (0,1)

#function 2 - tanh function
# precisely can be denoted as g(Z) = tanh(Z)

def tanh(x):
    numerator = exp(x) - exp(-x)
    denominator = exp(x) + exp(x)
    result = numerator/denominator
    return result
# best suitable for symmeteric cases
    
# function 3 - ReLu

def relu(x):
    ReLu = max(0,x)
    return ReLu

#range - [0,x)
# also if x >= 0 derivative of ReLu will be 1 else it will be 0

#function - 4 leaky_ReLu

def leaky_relu(x):
    leaky_ReLu = max(0.01*x, x)
    return leaky_ReLu
# we can change multiplying factor in order to increase precision. Here it is 0.01 it can be taken as 0.0001
# leaky_ReLu helps to gather some values which got vanished to 0 instead it provides them some negligible values

#function - 5 Softplus

#this function is introduced to overcome faults of leaky_ReLu
#Also known as Smoothened or Softened function (presence of smooth curve instead of joint in Relu or Leaky ReLu)

import numpy as np
def softplus(x):
    sfplus = np.log(1 + exp(x))
    return sfplus

#some more relation bw derivates of Activation function or bw Activation functions are included in other file.    
