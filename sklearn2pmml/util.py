from pandas import Index, Series

import numpy
import pandas

def cast(X, dtype):
	if isinstance(dtype, str) and dtype.startswith("datetime64"):
		func = lambda x: to_pydatetime(x, dtype)
		return dt_transform(X, func)
	else:
		if not hasattr(X, "astype"):
			X = numpy.asarray(X)
		return X.astype(dtype)

def eval_rows(X, func, dtype = object):
	if hasattr(X, "apply"):
		return X.apply(func, axis = 1)
	nrow = X.shape[0]
	Xt = numpy.empty(shape = (nrow, ), dtype = dtype)
	for i in range(0, nrow):
		Xt[i] = func(X[i])
	return Xt

def dt_transform(X, func):
	if hasattr(X, "applymap"):
		return X.applymap(func)
	shape = X.shape
	if len(shape) > 1:
		X = X.ravel()
	Xt = func(X)
	if isinstance(Xt, Index):
		Xt = Xt.values
	if len(shape) > 1:
		Xt = Xt.reshape(shape)
	return Xt

def to_pydatetime(X, dtype):
	Xt = pandas.to_datetime(X, yearfirst = True, origin = "unix")
	if hasattr(Xt, "dt"):
		Xt = Xt.dt
	if dtype == "datetime64[D]":
		Xt = Xt.floor("D")
	elif dtype == "datetime64[s]":
		Xt = Xt.floor("S")
	else:
		raise ValueError(dtype)
	return Xt.to_pydatetime()