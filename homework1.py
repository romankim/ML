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
    

def regressTestData():
    return getData('regress_test.txt')
    
    

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

"""
Problem 3.1
Implement ridge regression
"""
def ridgeRegression(X, Y, order, lam=0, verbose=True):
    pl.plot(X.T.tolist()[0],Y.T.tolist()[0], 'gs')
    phi = designMatrix(X, order)
    w   = np.dot( np.dot( np.linalg.inv( lam * np.identity(order) + np.dot(phi.T, phi) ), phi.T ), Y)
    if verbose:
        pts = [[p] for p in pl.linspace(min(X), max(X), 100)]
        Yp = pl.dot(w.T, designMatrix(pts, order).T)
         
        pl.plot(pts, Yp.tolist()[0])
    
    return w

def plot(X, Y, w, figurenum=1, designMatrix=designMatrix):
    pl.figure(figurenum)
    pl.plot(X.T.tolist()[0],Y.T.tolist()[0], 'gs')
    order = len(w)
    pts = [[p] for p in pl.linspace(min(X), max(X), 100)]
    Yp = pl.dot(w.T, designMatrix(pts, order).T)
     
    pl.plot(pts, Yp.tolist()[0])

"""
Problem 3.2
"""
def crossValidation():
    Xd, Yd  =  regressTrainData()
    Xv, Yv  =  regressValidateData()
    Xt, Yt  =  regressTestData()
    
    order   =  4
    #lam     =    np.e ** -18
    #lam     =    0
    lam     =    0
    w       =    ridgeRegression(Xd, Yd, order, lam)
    
    print "Training Data SSE\t M=%d, lam=%f" % (order, lam)
    sse_d   =    computeSSE(w, Xd, Yd, order, verbose=True)
    print "Valdiation Data SSE\t M=%d, lam=%f" % (order, lam)
    sse_v   =    computeSSE(w, Xv, Yv, order, verbose=True)
    print "Testing Data SSE\t M=%d, lam=%f" % (order, lam)
    sse_t   =    computeSSE(w, Xt, Yt, order, verbose=True)
    
def findBestRegularzation():
    Xd, Yd  =  regressTrainData()
    Xv, Yv  =  regressValidateData()
    Xt, Yt  =  regressTestData()

    orders       =   np.arange(1, 11, 1, dtype=int)
    lamdas       =   np.arange(0.0, 3.0, 0.01, dtype=float)
    bestSSE      =   np.inf
    bestorder    =   0.0
    bestlam      =   0.0
    bestweights  = []
    
    
    for order, lam in [(o,l) for o in orders for l in lamdas]:
        w       =    ridgeRegression( Xd, Yd, order, lam, verbose=False)
        sse_v   =    computeSSE(w, Xv, Yv, order, verbose=False)    
        
        if sse_v < bestSSE:
            bestorder, bestlam   = order, lam
            bestSSE              =  sse_v
            bestweights          =  w
    
    print "test data"
    plot(Xd, Yd, bestweights, 1)
    print "validation data"
    plot(Xv, Yv, bestweights, 2)
    
    print "M = {}, Lamda = {}, w = {}, SSE = {}".format(bestorder, bestlam, str(bestweights.flatten()), bestSSE )
    
    return bestorder, bestlam, bestweights
        
    


######################## Part 2 ###########################################


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
    
    goal, w = gd.gradientDescentNumerical( computeSSE, weights,thold=0.0001, **kwargs)
    print( "goal:{:.8f}\t coord:{}".format(goal, str(w)))
    
def testProblemThree():
    guess1 = np.array([ 1.0, 7.0, -25.0, 15.0 ])
    guess2 = np.zeros(4)
    
    testProblemThreeHelper(guess1)
    testProblemThreeHelper(guess2)

def testProblemFour():
    X,Y         =   bishopCurveData()
    order       =   8
    
    weights = regressionPlot(X,Y, order, designMatrix=designSineMatrix)
    
    print weights.flatten()
    computeSSE(weights, X, Y, order, verbose=True, designMatrix = designSineMatrix)
    
    
    
    
    

    
######################## Part 3 ###########################################    
    
# Ridge Regression
def test3_1():
    X, Y    =    bishopCurveData()
    order   =    10
    lam     =    np.e ** -18
    #lam     =    0
    #lam     =    1
    w       =    ridgeRegression(X, Y, order, lam)

    
    print "Ridge regresion weights: {}".format(str(w.flatten()))
    
    
    
def test3_2():
    crossValidation()
    
def test3_2_model_selection():
    return findBestRegularzation()
    
m,l,w = test3_2_model_selection()
