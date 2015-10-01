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
def ridgeRegression(X, Y, order, lam=0, designMatrix=designMatrix, verbose=True):
    phi     =  designMatrix(X, order)
    phiTphi =  np.dot(phi.T, phi)
    w   = np.dot( np.dot( np.linalg.inv( lam * np.identity(phiTphi.shape[0]) + phiTphi  ), phi.T ), Y)
    if verbose:
        pl.plot(X.T.tolist()[0],Y.T.tolist()[0], 'gs')
        pts = [[p] for p in pl.linspace(min(X), max(X), 100)]
        Yp = pl.dot(w.T, designMatrix(pts, order).T)
         
        pl.plot(pts, Yp.tolist()[0])
    
    return w

def plot(X, Y, w, figurenum=1, designMatrix=designMatrix, titlestring=None):
    pl.figure(figurenum)
    pl.plot(X.T.tolist()[0],Y.T.tolist()[0], 'gs')
    order = len(w)
    pts = [[p] for p in pl.linspace(min(X), max(X), 100)]
    Yp = pl.dot(w.T, designMatrix(pts, order).T)
     
    pl.plot(pts, Yp.tolist()[0])
    
    if titlestring is not None:
        pl.title(titlestring)

    
def getGrad(X, Y, w, designmMatrix=designMatrix) :
    if ( len(w.shape) == 1 ):
        w = w[:, np.newaxis]
    order   =  len(w.flatten())
    phi     =  designMatrix(X, order)
    err     =  Y - np.dot(phi, w)
    grad    =  -2.0 * np.dot(phi.T, err)
    
    return grad.flatten(0)


"""
Problem 3.3
"""
def blogTrainData():
    Xd = pl.loadtxt(r'BlogFeedback_data/x_train.csv', dtype=float, delimiter=",")
    Yd = pl.loadtxt(r'BlogFeedback_data/y_train.csv', dtype=float, delimiter=",")
    return Xd, Yd
def blogValData():
    Xv = pl.loadtxt(r'BlogFeedback_data/x_val.csv', dtype=float, delimiter=",")
    Yv = pl.loadtxt(r'BlogFeedback_data/y_val.csv', dtype=float, delimiter=",")
    return Xv, Yv
def blogTestData():
    Xt = pl.loadtxt(r'BlogFeedback_data/x_test.csv', dtype=float, delimiter=",")
    Yt = pl.loadtxt(r'BlogFeedback_data/y_test.csv', dtype=float, delimiter=",")
    return Xt, Yt

"""
Scaling methods due to https://en.wikipedia.org/wiki/Feature_scaling
Given an input of 1-D array, rescale it and return
"""
def rescaling(X):
    mn   =   np.min(X)
    mx   =   np.max(X)
    X    =   (X - mn) / (mx - mn)
    return X
def standarize(X):
    std  =   np.std(X)
    mean =   np.mean(X)
    X    =   (X-mean)/std
    return X
def designMatrixSelf(X, order=0):
    return X
"""
Problem 3.3
Finds the best regularization parameters..
"""    
def blogRegression(cluster=False):
    # indices 3, 8, 13, 18,23 are highly volatile
    Xd, Yd  =  blogTrainData()
    Xv, Yv  =  blogValData()
    Xt, Yt  =  blogTestData()

    lamdas       =   np.arange(0.01, 3.01, 0.1, dtype=float)
    bestSSE      =   np.inf
    bestlam      =   0.0
    bestweights  =   []
    """
    Dear Joohun,
    Please just ignore '0' that I am passing into the functions below.
    """
    
    for lam in lamdas:
        w       =    ridgeRegression( Xd, Yd, 0, lam, designMatrix=designMatrixSelf, verbose=False)
        sse_v   =    computeSSE(w, Xv, Yv, 0, verbose=False, designMatrix=designMatrixSelf)
        
        if sse_v < bestSSE:
            bestlam         =  lam
            bestSSE         =  sse_v
            bestweights     =  w
    
    if (not cluster) :
        print "SSE on Training Data:   %f" % computeSSE(bestweights, Xd, Yd, 0, verbose=False, designMatrix=designMatrixSelf)
        print "SSE on Validation Data: %f" % computeSSE(bestweights, Xv, Yv, 0, verbose=False, designMatrix=designMatrixSelf)
        print "SSE on Test Data:       %f" % computeSSE(bestweights, Xv, Yt, 0, verbose=False, designMatrix=designMatrixSelf)
        
        print "Optimum: Lamda = {}, w = {}, SSE = {}".format(bestlam, str(bestweights.flatten()), bestSSE )
    else :
        f = open('867_ps1_3_3.txt', 'wb')
        f.write("SSE on Training Data:   %f" % computeSSE(bestweights, Xd, Yd, 0, verbose=False, designMatrix=designMatrixSelf))
        f.write("SSE on Validation Data: %f" % computeSSE(bestweights, Xv, Yv, 0, verbose=False, designMatrix=designMatrixSelf))
        f.write("SSE on Test Data:       %f" % computeSSE(bestweights, Xv, Yt, 0, verbose=False, designMatrix=designMatrixSelf))
        f.write("Optimum: Lamda = {}, w = {}, SSE = {}".format(bestlam, str(bestweights.flatten()), bestSSE ))
        
        f.close()
        
    
    return bestlam, bestweights



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


"""
Problem 3.2
"""
def findBestRegularzation():
    Xd, Yd  =  regressTrainData()
    Xv, Yv  =  regressValidateData()
    Xt, Yt  =  regressTestData()

    orders       =   np.arange(1, 11, 1, dtype=int)
    lamdas       =   np.arange(0.0, 3.0, 0.01, dtype=float)
    bestSSE      =   np.inf
    bestorder    =   0.0
    bestlam      =   0.0
    bestweights  =   []
    
    
    for order, lam in [(o,l) for o in orders for l in lamdas]:
        w       =    ridgeRegression( Xd, Yd, order, lam, verbose=False)
        sse_v   =    computeSSE(w, Xv, Yv, order, verbose=False)
        
        if sse_v < bestSSE:
            bestorder, bestlam   = order, lam
            bestSSE              =  sse_v
            bestweights          =  w
    
    plot(Xd, Yd, bestweights, 1, titlestring="Ridge Regression on Training Data")
    plot(Xt, Yt, bestweights, 2, titlestring="Ridge Regression on Test Data")
    plot(Xv, Yv, bestweights, 3, titlestring="Ridge Regression on Validation Data")
    
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
    
    
        
    grad       = getGrad(X, Y, weights) 
    numergradv = gd.finiteDifference(computeSSE, weights.flatten(), **{ 'X':X, 'Y':Y, 'order':order, 'verbose':False})
    print grad.flatten()    
    print numergradv
    

    

def testProblemThreeHelper(guess):
    X, Y    =   bishopCurveData()
    order   =   4
    weights =   guess
    kwargs  =   { 'X':X, 'Y':Y, 'order':order, 'verbose':False}
    
    def getGradXY(w):
        X, Y = bishopCurveData()
        return getGrad(X,Y,w)
        
    
    goal, w = gd.gradientDescent( computeSSE, getGradXY, weights,thold=0.0001, **kwargs)
    print( "goal:{:.8f}\t coord:{}".format(goal, str(w)))
    
    return goal, w
    
def testProblemThree():
    Joo,Hun = bishopCurveData()    
    
    guess1  = np.array([ 1.0, 7.0, -25.0, 15.0 ])
    guess2  = np.zeros(4)
    
    goal_good, w_good = testProblemThreeHelper(guess1)
    goal_bad,  w_bad  = testProblemThreeHelper(guess2)
    
    plot(Joo,Hun, w_good[:,np.newaxis], 1, titlestring="Gradient Descent with Good Initial Guess")
    plot(Joo,Hun, w_bad[:,np.newaxis] , 2, titlestring="Gradient Descent with Bad Initial Guess")
    
    

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
    
def test3_3():
    blogRegression(cluster=True)
    

test3_3()
