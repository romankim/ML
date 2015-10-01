# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 17:37:02 2015

@author: Dongyoung
"""

"""
1. Implement a general Gradient Descent
2. Test the grad descent on a known function
3. 
"""
import numpy as np
from scipy.optimize import fmin_bfgs


"""
Problem 1.1
implements gradient descent

@param function f       target function we are trying to optimize
@param function gradf   gradient function of the target function
@param [float]  guess   initial guess
@param float    step    the learning rate
@param float    thold   thold difference in target function in two successive steps

@return float, [float]   minimum scalar value, and the correspondin coordnates

"""
def gradientDescent(f, gradf, guess, step=0.001, thold=0.00001, **kwargs) :
    coord  =  np.array(guess)
    obj    =  f(coord, **kwargs)
    diff   =  np.inf
    niter  =  0
        
    while (diff > thold) :
        delta   =  -step * np.array(gradf(coord))
        coord   =  coord + delta
        newobj  =  f(coord, **kwargs)
        diff    =  obj - newobj
        obj     =  newobj
        niter  += 1
    
    print "Gradient Descent"
    print "Number of function calls : %d" % niter
    
    return obj, coord
    
def gradientDescentNumerical(f, guess, step=0.001, thold=0.00001, **kwargs) :
    coord  =  np.array(guess)
    obj    =  f(coord, **kwargs)
    diff   =  np.inf
    niter  =  0
        
    while (diff > thold) :
        grad    =  np.array(finiteDifference(f, coord, **kwargs))
        delta   =  -step * grad
        coord   =  coord + delta
        newobj  =  f(coord, **kwargs)
        diff    =  obj - newobj
        obj     =  newobj
        niter  += 1
    
    print "Gradient Descent"
    print "Number of function calls : %d" % niter
    print"Gradient: %s" % str(grad.flatten())
    
    return obj, coord    

"""
Problem 1.2
"""
def quadraticBowl(l) :
    x = l[0]
    y = l[1]
    return (x**2.0 + y**2.0)
    
"""
Problem 1.2
"""    
def gradQuadraticBowl(l) :
    x = l[0]
    y = l[1]
    return [2.0*x, 2.0*y]


"""
Problem 1.2
tests the grad function on a Quadratic bowl
"""
def testQuad() :            
    x     = 100.0
    y     = 3.0
    step  = 0.1
    thold = 0.0015
    
    goal, coord = gradientDescent(f=quadraticBowl, gradf=gradQuadraticBowl, \
                    guess=[x, y], step=step, thold=thold )
                    
    print( "goal:{:.8f}\t coord:({:.8f},{:.8f})".format(goal, coord[0], coord[1]))




"""
Problem 1.3
calculates the numerical gradient
    
@param  function    f        target function
@param  [float]     point    inputs point to the function

@return [float]     gradient vector
"""
def finiteDifference(f, point, delta=0.0001, **kwargs):
    gradv =  np.zeros(shape=(len(point),))
    
    for d in xrange(len(point)) :
        vdelta      =    np.zeros(shape=(len(point),))
        vdelta[d]   =    delta
        fh          =    f((np.array(point) + vdelta), **kwargs)
        fl          =    f((np.array(point) - vdelta), **kwargs)
        pd          =    (fh-fl)/(2*delta)
        gradv[d]    =    pd
    
    return gradv
    
def testFiniteDifference():
    f       = quadraticBowl
    point   = [3.0, 15.0]
    
    gradv   = finiteDifference(f, point)
    
    print gradv
    
    
"""
Problem 1.4
Compare my gradientDescent with another scipy numerical optimizer
"""

def testScipy() : 
    # Initial guess
    x      =  100.0
    y      =  3.0
    step   =  0.1
    
    result = fmin_bfgs(quadraticBowl, x0=[x,y], epsilon=step, full_output=True, disp=False)
        
    # unpacking
    xopt  = result[0]
    fopt  = result[1]
    calls = result[4]
    
    print "Gradient Descent"
    print "Number of function calls : %d" % calls
    print( "goal:{}\t coord:({},{})".format(fopt, xopt[0], xopt[1]))
    

#testQuad()
#testScipy()
#    
#testFiniteDifference()
