from collections import defaultdict
from datetime import datetime
from pandas import Series
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import column_or_1d
from sklearn2pmml.util import eval_rows

import numpy
import pandas
import warnings

def _regex_engine(pattern):
	try:
		import pcre
		return pcre.compile(pattern)
	except ImportError:
		warnings.warn("Perl Compatible Regular Expressions (PCRE) library is not available, falling back to built-in Regular Expressions (RE) library. Transformation results might not be reproducible between Python and PMML environments when using more complex patterns", Warning)
		import re
		return re.compile(pattern)

class Aggregator(BaseEstimator, TransformerMixin):

	def __init__(self, function):
		functions = ["min", "max", "mean"]
		if function not in functions:
			raise ValueError("Function {0} not in {1}".format(function, functions))
		self.function = function

	def fit(self, X, y = None):
		return self

	def transform(self, X):
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

	def fit(self, X, y = None):
		X = column_or_1d(X, warn = True)
		return self

	def transform(self, X):
		X = column_or_1d(X, warn = True)
		return pandas.cut(X, bins = self.bins, right = self.right, labels = self.labels, include_lowest = self.include_lowest)

class DurationTransformer(BaseEstimator, TransformerMixin):

	def __init__(self, year):
		if year < 1900:
			raise ValueError("Year {0} is earlier than 1900".format(year))
		self.epoch = datetime(year, 1, 1, tzinfo = None)

	def _to_duration(self, td):
		return td

	def fit(self, X, y = None):
		return self

	def transform(self, X):
		shape = X.shape
		if len(shape) > 1:
			X = X.ravel()
		Xt = pandas.to_timedelta(X - self.epoch)
		Xt = self._to_duration(Xt)
		if len(shape) > 1:
			Xt = Xt.reshape(shape)
		return Xt

class DaysSinceYearTransformer(DurationTransformer):

	def __init__(self, year):
		super(DaysSinceYearTransformer, self).__init__(year)

	def _to_duration(self, td):
		return (td.days).values

class SecondsSinceYearTransformer(DurationTransformer):

	def __init__(self, year):
		super(SecondsSinceYearTransformer, self).__init__(year)

	def _to_duration(self, td):
		return ((td.total_seconds()).values).astype(int)

class ExpressionTransformer(BaseEstimator, TransformerMixin):

	def __init__(self, expr):
		self.expr = expr

	def _eval_row(self, X):
		return eval(self.expr)

	def fit(self, X, y = None):
		return self

	def transform(self, X):
		func = lambda x: self._eval_row(x)
		Xt = eval_rows(X, func)
		if isinstance(Xt, Series):
			Xt = Xt.values
		return Xt.reshape(-1, 1)

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

	def fit(self, X, y = None):
		X = column_or_1d(X, warn = True)
		return self

	def transform(self, X):
		X = column_or_1d(X, warn = True)
		transform_dict = self._transform_dict()
		func = lambda k: transform_dict[k]
		if hasattr(X, "apply"):
			return X.apply(func)
		return numpy.vectorize(func)(X)

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

	def fit(self, X, y = None):
		return self

	def transform(self, X):
		transform_dict = self._transform_dict()
		func = lambda k: transform_dict[tuple(k)]
		if hasattr(X, "apply"):
			return X.apply(func, axis = 1)
		# See https://stackoverflow.com/a/3338368
		return numpy.array([func((numpy.squeeze(numpy.asarray(row))).tolist()) for row in X])

class PMMLLabelBinarizer(BaseEstimator, TransformerMixin):

	def fit(self, X, y = None):
		X = column_or_1d(X, warn = True)
		self.classes_ = numpy.unique(X[~pandas.isnull(X)])
		return self

	def transform(self, X):
		X = column_or_1d(X, warn = True)
		index = list(self.classes_)
		Xt = numpy.zeros((len(X), len(index)), dtype = numpy.int)
		for i, v in enumerate(X):
			if not pandas.isnull(v):
				Xt[i, index.index(v)] = 1
		return Xt

class PMMLLabelEncoder(BaseEstimator, TransformerMixin):

	def __init__(self, missing_values = None):
		self.missing_values = missing_values

	def fit(self, X, y = None):
		X = column_or_1d(X, warn = True)
		self.classes_ = numpy.unique(X[~pandas.isnull(X)])
		return self

	def transform(self, X):
		X = column_or_1d(X, warn = True)
		index = list(self.classes_)
		return numpy.array([self.missing_values if pandas.isnull(v) else index.index(v) for v in X])

class PowerFunctionTransformer(BaseEstimator, TransformerMixin):

	def __init__(self, power):
		if not isinstance(power, int):
			raise ValueError("Power {0} is not an integer".format(power))
		self.power = power

	def fit(self, X, y = None):
		return self

	def transform(self, X):
		return numpy.power(X, self.power)

class ConcatTransformer(BaseEstimator, TransformerMixin):

	def __init__(self, separator = ""):
		self.separator = separator

	def fit(self, X, y = None):
		return self

	def transform(self, X):
		func = lambda x: self.separator.join([str(v) for v in x])
		Xt = eval_rows(X, func)
		if isinstance(Xt, Series):
			Xt = Xt.values
		return Xt.reshape(-1, 1)

class MatchesTransformer(BaseEstimator, TransformerMixin):

	def __init__(self, pattern):
		self.pattern = pattern

	def fit(self, X, y = None):
		X = column_or_1d(X, warn = True)
		return self

	def transform(self, X):
		X = column_or_1d(X, warn = True)
		engine = _regex_engine(self.pattern)
		func = lambda x: bool(engine.search(x))
		return eval_rows(X, func)

class ReplaceTransformer(BaseEstimator, TransformerMixin):

	def __init__(self, pattern, replacement):
		self.pattern = pattern
		self.replacement = replacement

	def fit(self, X, y = None):
		X = column_or_1d(X, warn = True)
		return self

	def transform(self, X):
		X = column_or_1d(X, warn = True)
		engine = _regex_engine(self.pattern)
		func = lambda x: engine.sub(self.replacement, x)
		return eval_rows(X, func)

class SubstringTransformer(BaseEstimator, TransformerMixin):

	def __init__(self, begin, end):
		if begin < 0:
			raise ValueError("Begin position {0} is negative".format(begin))
		if end < begin:
			raise ValueError("End position {0} is smaller than begin position {1}".format(end, begin))
		self.begin = begin
		self.end = end

	def fit(self, X, y = None):
		X = column_or_1d(X, warn = True)
		return self

	def transform(self, X):
		X = column_or_1d(X, warn = True)
		func = lambda x: x[self.begin:self.end]
		return eval_rows(X, func)

class StringNormalizer(BaseEstimator, TransformerMixin):

	def __init__(self, function = None, trim_blanks = True):
		functions = ["lowercase", "uppercase"]
		if (function is not None) and (function not in functions):
			raise ValueError("Function {0} not in {1}".format(function, functions))
		self.function = function
		self.trim_blanks = trim_blanks

	def fit(self, X, y = None):
		return self

	def transform(self, X):
		if hasattr(X, "values"):
			X = X.values
		Xt = X.astype("U")
		# Transform
		if self.function == "lowercase":
			Xt = numpy.char.lower(Xt)
		elif self.function == "uppercase":
			Xt = numpy.char.upper(Xt)
		# Trim blanks
		if self.trim_blanks:
			Xt = numpy.char.strip(Xt)
		return Xt
