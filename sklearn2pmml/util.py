import numpy

def eval_rows(X, func, dtype = object):
	if hasattr(X, "apply"):
		return X.apply(func, axis = 1)
	nrow = X.shape[0]
	y = numpy.empty(shape = (nrow, ), dtype = dtype)
	for i in range(0, nrow):
		y[i] = func(X[i])
	return y
