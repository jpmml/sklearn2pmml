import numpy
import pandas

def eval_rows(X, func, dtype = object):
	if hasattr(X, "apply"):
		return X.apply(func, axis = 1)
	nrow = X.shape[0]
	Xt = numpy.empty(shape = (nrow, ), dtype = dtype)
	for i in range(0, nrow):
		Xt[i] = func(X[i])
	return Xt

def flat_transform(X, func):
	shape = X.shape
	if len(shape) > 1:
		X = X.ravel()
	Xt = func(X)
	if len(shape) > 1:
		Xt = Xt.reshape(shape)
	return Xt

def to_pydatetime(X, dtype):
	Xt = pandas.to_datetime(X, yearfirst = True, origin = "unix")
	if dtype == "datetime64[D]":
		Xt = Xt.floor("D")
	elif dtype == "datetime64[s]":
		Xt = Xt.floor("S")
	else:
		raise ValueError(dtype)
	return Xt.to_pydatetime()