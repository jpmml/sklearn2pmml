from collections import defaultdict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import column_or_1d

import numpy
import pandas

class Aggregator(BaseEstimator, TransformerMixin):

	def __init__(self, function):
		functions = ["min", "max", "mean"]
		if function not in functions:
			raise ValueError("Function {0} not in {1}".format(function, functions))
		self.function = function

	def fit(self, X, y = None):
		return self

	def transform(self, X, y = None):
		if self.function == "min":
			return numpy.amin(X, axis = 1) 
		elif self.function == "max":
			return numpy.amax(X, axis = 1)
		elif self.function == "mean":
			return numpy.mean(X, axis = 1)
		return X

class CutTransformer(BaseEstimator, TransformerMixin):

	def __init__(self, bins, right = True, labels = None, include_lowest = True):
		self.bins = bins
		self.right = right
		self.labels = labels
		self.include_lowest = include_lowest

	def fit(self, y):
		y = column_or_1d(y, warn = True)
		return self

	def transform(self, y):
		y = column_or_1d(y, warn = True)
		return pandas.cut(y, bins = self.bins, right = self.right, labels = self.labels, include_lowest = self.include_lowest)

class ExpressionTransformer(BaseEstimator, TransformerMixin):

	def __init__(self, expr):
		self.expr = expr

	def fit(self, X, y = None):
		return self

	def transform(self, X, y = None):
		return eval(self.expr)

class LookupTransformer(BaseEstimator, TransformerMixin):

	def __init__(self, mapping, default_value):
		if type(mapping) is not dict:
			raise ValueError("Input value to output value mapping is not a dict")
		for k, v in mapping.items():
			if k is None:
				raise ValueError("Key is None")
		self.mapping = mapping
		self.default_value = default_value

	def _transform_dict(self):
		transform_dict = defaultdict(lambda: self.default_value)
		transform_dict.update(self.mapping)
		return transform_dict

	def fit(self, y):
		y = column_or_1d(y, warn = True)
		return self

	def transform(self, y):
		y = column_or_1d(y, warn = True)
		transform_dict = self._transform_dict()
		func = lambda k: transform_dict[k]
		if hasattr(y, "apply"):
			return y.apply(func)
		return numpy.vectorize(func)(y)

class MultiLookupTransformer(LookupTransformer):

	def __init__(self, mapping, default_value):
		super(MultiLookupTransformer, self).__init__(mapping, default_value)
		length = -1
		for k, v in mapping.items():
			if type(k) is not tuple:
				raise ValueError("Key is not a tuple")
			if length == -1:
				length = len(k)
				continue
			if length != len(k):
				raise ValueError("Keys contain variable number of elements")

	def fit(self, Y):
		return self

	def transform(self, Y):
		transform_dict = self._transform_dict()
		func = lambda k: transform_dict[tuple(k)]
		if hasattr(Y, "apply"):
			return Y.apply(func, axis = 1)
		# See https://stackoverflow.com/a/3338368
		return numpy.array([func((numpy.squeeze(numpy.asarray(row))).tolist()) for row in Y])

class PMMLLabelBinarizer(BaseEstimator, TransformerMixin):

	def fit(self, y):
		y = column_or_1d(y, warn = True)
		self.classes_ = numpy.unique(y[~pandas.isnull(y)])
		return self

	def transform(self, y):
		y = column_or_1d(y, warn = True)
		index = list(self.classes_)
		Y = numpy.zeros((len(y), len(index)), dtype = numpy.int)
		for i, v in enumerate(y):
			if not pandas.isnull(v):
				Y[i, index.index(v)] = 1
		return Y

class PMMLLabelEncoder(BaseEstimator, TransformerMixin):

	def __init__(self, missing_value = None):
		self.missing_value = missing_value

	def fit(self, y):
		y = column_or_1d(y, warn = True)
		self.classes_ = numpy.unique(y[~pandas.isnull(y)])
		return self

	def transform(self, y):
		y = column_or_1d(y, warn = True)
		index = list(self.classes_)
		return numpy.array([self.missing_value if pandas.isnull(v) else index.index(v) for v in y])

class PowerFunctionTransformer(BaseEstimator, TransformerMixin):

	def __init__(self, power):
		if not isinstance(power, int):
			raise ValueError("Power {0} is not an integer".format(power))
		self.power = power

	def fit(self, X, y = None):
		return self

	def transform(self, X, y = None):
		return numpy.power(X, self.power)

class StringNormalizer(BaseEstimator, TransformerMixin):

	def __init__(self, function = None, trim_blanks = True):
		functions = ["lowercase", "uppercase"]
		if (function is not None) and (function not in functions):
			raise ValueError("Function {0} not in {1}".format(function, functions))
		self.function = function
		self.trim_blanks = trim_blanks

	def fit(self, X, y = None):
		return self

	def transform(self, X, y = None):
		if hasattr(X, "values"):
			X = X.values
		X = X.astype("U")
		if self.function == "lowercase":
			X = numpy.char.lower(X)
		elif self.function == "uppercase":
			X = numpy.char.upper(X)
		if self.trim_blanks:
			X = numpy.char.strip(X)
		return X
