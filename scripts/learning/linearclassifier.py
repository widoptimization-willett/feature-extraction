import numpy as np

"""
linear classification and verification
"""

def train_weights(l, X, Y):
	"""
	Parameters
	----------
	l : float
		label error coefficient
	X : array_like [[float, ...], ...]
		vector of feature vectors in training set
	Y : array_like [int, int, ...]
		vector of integer labels
	
	Returns
	-------
	theta_hat : array_like [float, float, ...]
		the model weights
	"""
	
	return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X) + l*np.identity(X.shape[1])), X.T), Y)

def predict(X, weights):
	return np.dot(X, weights)

def prediction_errors(Y, _Y):
	_Y_clipped = (_Y>0)*2-1
	return float(np.count_nonzero(Y-_Y_clipped))

def errors_for_class(c, Y, _Y):
	_Y_clipped = (_Y>0)*2-1
	errmask = (Y-_Y_clipped) != 0
	return float((errmask & (Y==c)).sum())

def find_optimal_weights(L, X_train, Y_train, X_verification, Y_verification):
	errors = [prediction_errors(Y_verification,
								predict(X_verification, train_weights(l, X_train, Y_train))) for l in L]

	best_l = L[np.argmin(errors)]
	best_weights = train_weights(best_l, X_train, Y_train)
	
	return (best_weights, best_l)
