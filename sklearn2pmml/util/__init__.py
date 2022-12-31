from pandas import Categorical, DataFrame, Index, Series
from sklearn.base import clone, BaseEstimator, TransformerMixin

import numpy
import pandas
import sys
import types

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

def fqn(obj):
	clazz = obj.__class__
	return ".".join([clazz.__module__, clazz.__name__])

def is_instance_attr(obj, name):
	if not hasattr(obj, name):
		return False
	if name.startswith("__") and name.endswith("__"):
		return False
	v = getattr(obj, name)
	if isinstance(v, (types.BuiltinFunctionType, types.BuiltinMethodType, types.FunctionType, types.MethodType)):
		return False
	# See https://stackoverflow.com/a/17735709/
	attr_type = getattr(type(obj), name, None)
	if isinstance(attr_type, property):
		return False
	return True

def get_instance_attrs(obj):
	names = dir(obj)
	names = [name for name in names if is_instance_attr(obj, name)]
	return names

def sizeof(obj, with_overhead = False):
	if with_overhead:
		return sys.getsizeof(obj)
	return obj.__sizeof__()

def deep_sizeof(obj, with_overhead = False, verbose = False):
	# Primitive type values
	if obj is None:
		return obj.__sizeof__()
	elif isinstance(obj, (int, float, str, bool, numpy.int64, numpy.float32, numpy.float64)):
		return obj.__sizeof__()
	# Iterables
	elif isinstance(obj, list):
		sum = sizeof([], with_overhead = with_overhead) # Empty list
		for v in obj:
			v_sizeof = deep_sizeof(v, with_overhead = with_overhead, verbose = False)
			sum += v_sizeof
		return sum
	elif isinstance(obj, tuple):
		sum = sizeof((), with_overhead = with_overhead) # Empty tuple
		for i, v in enumerate(obj):
			v_sizeof = deep_sizeof(v, with_overhead = with_overhead, verbose = False)
			sum += v_sizeof
		return sum
	# Numpy ndarrays
	elif isinstance(obj, numpy.ndarray):
		sum = sizeof(obj, with_overhead = with_overhead) # Array header
		sum += (obj.size * obj.itemsize) # Array content
		return sum
	# Reference type values
	else:
		qualname = fqn(obj)
		# Restrict the circle of competence to Scikit-Learn classes
		if not (qualname.startswith("_abc.") or qualname.startswith("sklearn.")):
			raise TypeError("The object (class {0}) is not supported ".format(qualname))
		sum = sizeof(object(), with_overhead = with_overhead) # Empty object
		names = get_instance_attrs(obj)
		if names:
			if verbose:
				print("| Attribute | `type(v)` | `deep_sizeof(v)` |")
				print("|---|---|---|")
			for name in names:
				v = getattr(obj, name)
				v_type = type(v)
				v_sizeof = deep_sizeof(v, with_overhead = with_overhead, verbose = False)
				sum += v_sizeof
				if verbose:
					print("| {} | {} | {} |".format(name, v_type, v_sizeof))
		return sum

class Reshaper(BaseEstimator, TransformerMixin):

	def __init__(self, newshape):
		self.newshape = newshape

	def fit(self, X, y = None):
		return self

	def transform(self, X):
		return X.reshape(self.newshape)

class Slicer(BaseEstimator, TransformerMixin):

	def __init__(self, start = None, stop = None, step = None):
		self.start = start
		self.stop = stop
		if (step is not None) and (step <= 0):
			raise ValueError("Step must be positive integer")
		self.step = step

	def fit(self, X, y = None):
		return self

	def transform(self, X):
		rows = slice(None, None)
		columns = slice(self.start, self.stop, self.step)
		if isinstance(X, DataFrame):
			return X.iloc[rows, columns]
		else:
			return X[rows, columns]
