from pandas import Categorical, DataFrame, Index, Series
from sklearn.base import clone, BaseEstimator, TransformerMixin

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

def common_dtype(X):
	if hasattr(X, "dtype"):
		return X.dtype
	elif hasattr(X, "dtypes"):
		dtypes = set(X.dtypes)
		if len(dtypes) != 1:
			raise ValueError(dtypes)
		return next(iter(dtypes))
	else:
		raise ValueError()

def ensure_1d(X):
	if isinstance(X, (Categorical, Series)):
		return X
	elif isinstance(X, DataFrame):
		columns = X.columns
		if len(columns) == 1:
			return X[columns[0]]
	X = numpy.asarray(X)
	shape = X.shape
	if len(shape) == 1:
		return X
	elif (len(shape) == 2) and (shape[1] == 1):
		return X.ravel()
	else:
		raise ValueError("Expected 1d array, got {0}d array of shape {1}".format(len(shape), str(shape)))

def eval_rows(X, func, dtype = object):
	if hasattr(X, "apply"):
		if isinstance(X, Series):
			return X.apply(func)
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

class Reshaper(BaseEstimator, TransformerMixin):

	def __init__(self, newshape):
		self.newshape = newshape

	def fit(self, X, y = None):
		return self

	def transform(self, X):
		return X.reshape(self.newshape)
