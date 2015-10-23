import numpy as np
from plotBoundary import *
from scipy.optimize import fmin_tnc # Scipy Gradient Descent
from sklearn.linear_model import SGDClassifier

# import your LR training code

# parameters

print '======Training======'
# load data from csv files


############### parameters ###########
name  = 'stdev4'
alpha = 0.01
############### parameters ###########




train = np.loadtxt('data/data_'+name+'_train.csv')
X = train[:,0:2]
Y = train[:,2:3]

# Carry out training.
# alpha is the regularzation term
clf = SGDClassifier(loss="log", penalty="l2", alpha=alpha)
clf.fit(X,Y)



# Define the predictLR(x) function, which uses trained parameters
def predictLR(x) :
    return clf.predict(x)[0]

# plot training results
plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Train')

print '======Validation======'
# load data from csv files
validate = np.loadtxt('data/data_'+name+'_validate.csv')
X = validate[:,0:2]
Y = validate[:,2:3]

# plot validation results
plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Validate')
