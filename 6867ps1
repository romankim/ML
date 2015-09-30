import pdb
import random
import pylab as pl
import numpy as np
import Gradient_descent as gd
from scipy.optimize import fmin_bfgs


"""
Problem 2.1
Returns the Phi matrix (as given in Bishop) for dataset X
"""
def designMatrix(X, order) :    
    columnIndex  = np.fromfunction(lambda i,j: j, (len(X), order), dtype = int)
    xMatrix      = np.tile(X, (order,))
    phi          = xMatrix ** columnIndex
    return phi
"""
Problem  2.1
Calculates the Bishop Equation 3.15
"""
def regressionFit(X, Y, phi):
    return np.dot( np.dot( np.linalg.inv( np.dot(phi.T, phi) ), phi.T ), Y)


"""
Problem 2.4
"""
def designSineMatrix(X, order) :
    columnIndex  = np.fromfunction(lambda i,j: j, (len(X), order), dtype = int)
    xMatrix      = np.tile(X, (order,))
    phi          = np.zeros(shape=xMatrix.shape) 
    
    for i in xrange( phi.shape[0] ) :
        for j in xrange( phi.shape[1] ) :
            phi[i][j] = np.sin(xMatrix[i][j] * 2.0 * np.pi * j) 
    
    phi[:,0] = np.ones( phi.shape[0] )
    
    return phi
    
    
        
    

# X is an array of N data points (one dimensional for now), that is, NX1
# Y is a Nx1 column vector of data values
# order is the order of the highest order polynomial in the basis functions
def regressionPlot(X, Y, order, designMatrix = designMatrix):
    assert(len(X) == len(Y))
    pl.plot(X.T.tolist()[0],Y.T.tolist()[0], 'gs')

    # You will need to write the designMatrix and regressionFit function

    # constuct the design matrix (Bishop 3.16), the 0th column is just 1s.
    phi = designMatrix(X, order)
    # compute the weight vector
    w = regressionFit(X, Y, phi)

    # produce a plot of the values of the function 
    pts = [[p] for p in pl.linspace(min(X), max(X), 100)]
    Yp = pl.dot(w.T, designMatrix(pts, order).T)
    pl.plot(pts, Yp.tolist()[0])
    
    return w

  
    


def getData(name):
    data = pl.loadtxt(name)
    # Returns column matrices
    X = data[0:1].T
    Y = data[1:2].T
    return X, Y

def bishopCurveData():
    # y = sin(2 pi x) + N(0,0.3),
    return getData('curvefitting.txt')

def regressTrainData():
    return getData('regress_train.txt')

def regressValidateData():
    return getData('regress_validate.txt')

#problem 2.2
"""
@params
"""
def computeSSE(w, X, Y, order, verbose=True, designMatrix = designMatrix):
    if len(w.shape) == 1 :
        w = w[:, np.newaxis]
    
    phi = designMatrix(X, order)
        
    SSE = np.dot( (Y - np.dot(phi, w)).T, (Y - np.dot(phi, w)) )
    SSE = SSE.flatten()[0]
    
    if verbose:
        print('SSE: ', SSE)
    
    return SSE
    



######################## Tests ###########################################


def testProblemOne(order=4):
    X,Y = bishopCurveData()
    
    weights = regressionPlot(X, Y, order)
    weights = weights.flatten()
    goal    = computeSSE(weights, X, Y, order, verbose=False)
    print( "goal:{:.8f}\t coord:{}".format(goal, str(weights)))
    
def testProblemTwo():
    X, Y    =   bishopCurveData()
    order   =   4
    weights =   regressionPlot(X,Y,order)
    
    computeSSE(weights, X, Y, order)
    
    gradv = gd.finiteDifference(computeSSE, weights.flatten(), **{ 'X':X, 'Y':Y, 'order':order, 'verbose':False})
    print gradv
    

def testProblemThreeHelper(guess):
    X, Y    =   bishopCurveData()
    order   =   4
    weights =   guess
    kwargs  =   { 'X':X, 'Y':Y, 'order':order, 'verbose':False}
    
    goal, w = gd.gradientDescentNumerical( computeSSE, weights, **kwargs )
    print( "goal:{:.8f}\t coord:{}".format(goal, str(w)))
    
def testProblemThree():
    guess1 = np.array([ 1.0, 7.0, -25.0, 15.0 ])
    guess2 = np.zeros(4)
    
    testProblemThreeHelper(guess1)
    testProblemThreeHelper(guess2)

def testProblemFour():
    X,Y         =   bishopCurveData()
    order       =   2
    
    weights = regressionPlot(X,Y, order, designMatrix=designSineMatrix)
    
    print weights.flatten()
    computeSSE(weights, X, Y, order, verbose=True, designMatrix = designSineMatrix)
    
testProblemFour()
