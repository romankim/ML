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
from scipy.stats import multivariate_normal

"""
implements gradient descent

@param function f       target function we are trying to optimize
@param function gradf   gradient function of the target function
@param [float]  guess   initial guess
@param float    step    the learning rate
@param float    thold   thold difference in target function in two successive steps

@return float, [float]   minimum scalar value, and the correspondin coordnates

"""
def gradientDescent(f, gradf, guess, step=0.001, thold=0.00001) :
    coord  =  np.array(guess)
    obj    =  f(*coord)
    diff   =  np.inf
        
    while (diff > thold) :
        delta   =  -step * np.array(gradf(*coord))
        coord   =  coord + delta
        newobj  =  f(*coord)
        diff    =  obj - newobj
        obj     =  newobj
    
    return obj, coord
    


def quadraticBowl(x,y) : 
    return (x**2.0 + y**2.0)

def gradQuadraticBowl(x,y) :
    return [2.0*x, 2.0*y]


"""
tests the grad function on a multivaraite gaussian normal
"""
def testQuad() :            
    obj, coord = gradientDescent(f=quadraticBowl, gradf=gradQuadraticBowl, \
                    guess=[2.0, 3.0], step=0.1, thold = 0.0 )
                    
    print( "obj:{:.3f}\t coord:({:.3f},{:.3f})".format(obj, coord[0], coord[1]))
    

"""
calculates the numerical gradient

@param  function    f        target function
@param  [float]     point    inputs point to the function

@return [float]     gradient vector
"""
def finiteDifference(f, point, delta=0.0001):
    gradv =  np.zeros(shape=(len(point),))
    
    for d in xrange(len(point)) :
        vdelta      =    np.zeros(shape=(len(point),))
        vdelta[d]   =    delta
        fh          =    f(*(np.array(point) + vdelta))
        fl          =    f(*(np.array(point) - vdelta))
        pd          =    (fh-fl)/(2*delta)
        gradv[d]    =    pd
    
    return gradv

    
    
    
