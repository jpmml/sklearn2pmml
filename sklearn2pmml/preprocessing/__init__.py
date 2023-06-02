from collections import defaultdict
try:
	from collections.abc import Hashable
except ImportError:
	from collections import Hashable
from datetime import datetime
from io import StringIO
from pandas import DataFrame, Series
from scipy.interpolate import BSpline
from scipy.sparse import lil_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn2pmml.util import cast, dt_transform, ensure_1d, ensure_def, eval_rows, to_expr_func, Expression

import numpy
import pandas
import types
import warnings

def _regex_engine(pattern):
	try:
		import pcre
		return pcre.compile(pattern)
	except ImportError:
		warnings.warn("Perl Compatible Regular Expressions (PCRE) library is not available, falling back to built-in Regular Expressions (RE) library. Transformation results might not be reproducible between Python and PMML environments when using more complex patterns", Warning)
		import re
		return re.compile(pattern)

def _col2d(X):
	if isinstance(X, Series):
		X = X.values
	return X.reshape(-1, 1)

def _int(X):
	if numpy.isscalar(X):
		return int(X)
	else:
		if isinstance(X, Series):
			X = X.values
		return X.astype(int)

class Aggregator(BaseEstimator, TransformerMixin):
	"""Aggregate continuous data."""

	def __init__(self, function):
		functions = ["min", "max", "sum", "prod", "product", "mean", "avg"]
		if function not in functions:
			raise ValueError("Function {0} not in {1}".format(function, functions))
		self.function = function

	def fit(self, X, y = None):
		return self

	def transform(self, X):
		if self.function == "min":
			Xt = numpy.nanmin(X, axis = 1) 
		elif self.function == "max":
			Xt = numpy.nanmax(X, axis = 1)
		elif self.function == "sum":
			Xt = numpy.nansum(X, axis = 1)
		elif self.function == "prod" or self.function == "product":
			Xt = numpy.nanprod(X, axis = 1)
		elif self.function == "mean" or self.function == "avg":
			Xt = numpy.nanmean(X, axis = 1)
		else:
			raise ValueError(self.function)
		return _col2d(Xt)

class BSplineTransformer(BaseEstimator, TransformerMixin):

	def __init__(self, bspline):
		if not isinstance(bspline, BSpline):
			raise TypeError("The spline object is not an instance of {0}".format(BSpline.__name__))
		self.bspline = bspline

	def fit(self, X, y = None):
		X = ensure_1d(X)
		return self

	def transform(self, X):
		X = ensure_1d(X)
		return self.bspline(X)

class CastTransformer(BaseEstimator, TransformerMixin):
	"""Change data type."""

	def __init__(self, dtype):
		if isinstance(dtype, str) and dtype.startswith("datetime64"):
			dtypes = ["datetime64[D]", "datetime64[s]"]
			if dtype not in dtypes:
				raise ValueError("Temporal data type {0} not in {1}".format(dtype, dtypes))
		self.dtype = dtype

	def fit(self, X, y = None):
		return self

	def transform(self, X):
		return cast(X, self.dtype)

class CutTransformer(BaseEstimator, TransformerMixin):
	"""Bin continuous data to categorical."""

	def __init__(self, bins, right = True, labels = None, include_lowest = True):
		self.bins = bins
		self.right = right
		self.labels = labels
		self.include_lowest = include_lowest

	def fit(self, X, y = None):
		X = ensure_1d(X)
		return self

	def transform(self, X):
		X = ensure_1d(X)
		Xt = pandas.cut(X, bins = self.bins, right = self.right, labels = self.labels, include_lowest = self.include_lowest)
		if Xt.dtype == "category":
			Xt = numpy.asarray(Xt)
		return _col2d(Xt)

class DataFrameConstructor(BaseEstimator, TransformerMixin):

	def __init__(self, columns, dtype):
		self.columns = columns
		self.dtype = dtype

	def fit(self, X, y = None):
		return self

	def transform(self, X):
		return DataFrame(X, columns = self.columns, dtype = self.dtype)

class DurationTransformer(BaseEstimator, TransformerMixin):
	"""Calculate time difference."""

	def __init__(self, year):
		if year < 1900:
			raise ValueError("Year {0} is earlier than 1900".format(year))
		self.year = year
		self.epoch = datetime(year, 1, 1, tzinfo = None)

	def _to_duration(self, td):
		raise NotImplementedError()

	def fit(self, X, y = None):
		return self

	def transform(self, X):
		def to_int_duration(X):
			duration = self._to_duration(pandas.to_timedelta(X - self.epoch))
			return _int(duration)
		return dt_transform(X, to_int_duration)

class DaysSinceYearTransformer(DurationTransformer):
	"""Calculate the number of days since the epoch."""

	def __init__(self, year):
		super(DaysSinceYearTransformer, self).__init__(year)

	def _to_duration(self, td):
		return td.days

class SecondsSinceYearTransformer(DurationTransformer):
	"""Calculate the number of seconds since the epoch."""

	def __init__(self, year):
		super(SecondsSinceYearTransformer, self).__init__(year)

	def _to_duration(self, td):
		return td.total_seconds()

class SecondsSinceMidnightTransformer(BaseEstimator, TransformerMixin):
	"""Calculate the number of seconds since midnight."""

	def __init__(self):
		pass

	def _to_duration(self, td):
		return td.seconds

	def fit(self, X, y = None):
		return self

	def transform(self, X):
		def to_int_duration(X):
			dt = pandas.to_datetime(X)
			duration = self._to_duration(dt - dt.normalize())
			return _int(duration)
		return dt_transform(X, to_int_duration)

class ExpressionTransformer(BaseEstimator, TransformerMixin):
	"""Transform data using a Python expression.

	Parameters:
	----------
	map_missing_to: scalar, optional
		The return value when any of the expression arguments is missing.

	default_value: scalar, optional
		The return value when the expression result is missing.

	invalid_value_treatment: string
		The action to take when the evaluation of the expression raises an error.
	"""

	def __init__(self, expr, map_missing_to = None, default_value = None, invalid_value_treatment = None, dtype = None):
		if not isinstance(expr, (str, Expression)):
			raise TypeError()
		self.expr = expr
		self.map_missing_to = map_missing_to
		self.default_value = default_value
		invalid_value_treatments = ["return_invalid", "as_missing"]
		if (invalid_value_treatment is not None) and (invalid_value_treatment not in invalid_value_treatments):
			raise ValueError("Invalid value treatment {0} not in {1}".format(invalid_value_treatment, invalid_value_treatments))
		self.invalid_value_treatment = invalid_value_treatment
		self.dtype = dtype

	def fit(self, X, y = None):
		return self

	def transform(self, X):
		expr_func = to_expr_func(self.expr)

		def _eval_row(x):
			# x is array-like (row vector)
			if (self.map_missing_to is not None) and ((pandas.isnull(x)).any()):
				return self.map_missing_to
			try:
				xt = expr_func(x)
			except ArithmeticError as ae:
				if self.invalid_value_treatment == "return_invalid":
					raise ae
				elif self.invalid_value_treatment == "as_missing":
					xt = None
				else:
					pass
			# xt is scalar
			if (self.default_value is not None) and (pandas.isnull(xt)):
				return self.default_value
			return xt

		# Evaluate in PMML compatibility mode
		with numpy.errstate(divide = "raise"):
			Xt = eval_rows(X, _eval_row)
		if self.dtype is not None:
			Xt = cast(Xt, self.dtype)
		return _col2d(Xt)

class DateTimeFormatter(BaseEstimator, TransformerMixin):
	"""Formats dates, times and datetimes according to a pattern. Analogous to C's strftime() function.

	Parameters:
	----------
	pattern: string
		A POSIX-compliant formatting pattern.
	"""

	def __init__(self, pattern):
		self.pattern = pattern

	def _strftime(self, x):
		return x.strftime(self.pattern)

	def fit(self, X):
		return self

	def transform(self, X):
		X = ensure_1d(X)
		func = lambda x: self._strftime(x)
		Xt = eval_rows(X, func)
		return _col2d(Xt)

class NumberFormatter(BaseEstimator, TransformerMixin):
	"""Formats numbers according to a pattern. Analogous to C's printf() function.

	Parameters:
	----------
	pattern: string
		A POSIX-compliant formatting pattern.
	"""

	def __init__(self, pattern):
		self.pattern = pattern

	def _printf(self, x):
		with StringIO() as buffer:
			print(self.pattern % (x), sep = "", end = "", file = buffer)
			return buffer.getvalue()

	def fit(self, X):
		return self

	def transform(self, X):
		X = ensure_1d(X)
		func = lambda x: self._printf(x)
		Xt = eval_rows(X, func)
		return _col2d(Xt)

class LookupTransformer(BaseEstimator, TransformerMixin):
	"""Re-map 1D categorical data.

	If the mapping is not found, returns `default_value`.

	See also:
	--------
	FilterLookupTransformer

	"""

	def __init__(self, mapping, default_value):
		if type(mapping) is not dict:
			raise TypeError("Input value to output value mapping is not a dict")
		k_type = None
		v_type = None
		for k, v in mapping.items():
			if k is None:
				raise ValueError("Key is None")
			if k_type is None:
				k_type = type(k)
			else:
				if type(k) != k_type:
					raise TypeError("Key is not a {0}".format(k_type.__name__))
			if v is None:
				continue
			if v_type is None:
				v_type = type(v)
			else:
				if type(v) != v_type:
					raise TypeError("Value is not a {0}".format(v_type.__name__))
		self.mapping = mapping
		if default_value is not None:
			if v_type is not None:
				if type(default_value) != v_type:
					raise TypeError("Default value is not a {0}".format(v_type.__name__))
		self.default_value = default_value

	def _transform_dict(self):
		transform_dict = defaultdict(lambda: self.default_value)
		transform_dict.update(self.mapping)
		return transform_dict

	def fit(self, X, y = None):
		X = ensure_1d(X)
		return self

	def transform(self, X):
		X = ensure_1d(X)
		transform_dict = self._transform_dict()
		func = lambda k: transform_dict[k]
		Xt = eval_rows(X, func)
		return _col2d(Xt)

class FilterLookupTransformer(LookupTransformer):
	"""Selectively re-map 1D categorical data.

	If the mapping is not found, returns the original value unchanged.

	See also:
	--------
	LookupTransformer

	"""

	def __init__(self, mapping):
		super(FilterLookupTransformer, self).__init__(mapping, default_value = None)
		for k, v in mapping.items():
			if v is None:
				raise ValueError("Value is None")
			if type(k) != type(v):
				raise TypeError("Key and Value type mismatch")

	def _transform_dict(self):
		return self.mapping

	def fit(self, X, y = None):
		return self

	def transform(self, X):
		X = ensure_1d(X)
		transform_dict = self._transform_dict()
		func = lambda k: transform_dict[k] if k in transform_dict else k
		Xt = eval_rows(X, func)
		return _col2d(Xt)

def _deeptype(t):
	return tuple([type(e) for e in t])

class MultiLookupTransformer(LookupTransformer):
	"""Re-map multidimensional categorical data."""

	def __init__(self, mapping, default_value):
		super(MultiLookupTransformer, self).__init__(mapping, default_value)
		k_type = None
		for k, v in mapping.items():
			if not isinstance(k, tuple):
				raise TypeError("Key is not a tuple")
			if k_type is None:
				k_type = _deeptype(k)
			else:
				if _deeptype(k) != k_type:
					raise TypeError("Key is not a tuple of {0}", ", ".join(e.__name__ for e in k_type))

	def fit(self, X, y = None):
		return self

	def transform(self, X):
		transform_dict = self._transform_dict()
		# See https://stackoverflow.com/a/3460747
		# See https://stackoverflow.com/a/3338368
		func = lambda k: transform_dict[tuple(k) if isinstance(k, Hashable) else tuple(numpy.squeeze(numpy.asarray(k)))]
		Xt = eval_rows(X, func)
		return _col2d(Xt)

def _make_index(values):
	result = {}
	for i, v in enumerate(list(values)):
		result[v] = i
	if len(result) != len(values):
		raise ValueError()
	return result

class PMMLLabelBinarizer(BaseEstimator, TransformerMixin):
	"""Binarize categorical data in a missing value-aware way."""

	def __init__(self, sparse_output = False):
		self.sparse_output = sparse_output

	def fit(self, X, y = None):
		X = ensure_1d(X)
		self.classes_ = numpy.unique(X[~pandas.isnull(X)])
		return self

	def transform(self, X):
		X = ensure_1d(X)
		mapping = _make_index(self.classes_)
		if self.sparse_output:
			Xt = lil_matrix((len(X), len(mapping)), dtype = int)
		else:
			Xt = numpy.zeros((len(X), len(mapping)), dtype = int)
		for i, v in enumerate(X):
			if (pandas.notnull(v)) and (v in mapping):
				Xt[i, mapping[v]] = 1
		if self.sparse_output:
			Xt = Xt.tocsr()
		return Xt

class PMMLLabelEncoder(BaseEstimator, TransformerMixin):
	"""Encode categorical data in a missing value-aware way."""

	def __init__(self, missing_values = None):
		self.missing_values = missing_values

	def fit(self, X, y = None):
		X = ensure_1d(X)
		self.classes_ = numpy.unique(X[~pandas.isnull(X)])
		return self

	def transform(self, X):
		X = ensure_1d(X)
		mapping = _make_index(self.classes_)
		Xt = numpy.array([self.missing_values if pandas.isnull(v) else mapping.get(v, self.missing_values) for v in X])
		return _col2d(Xt)

class PowerFunctionTransformer(BaseEstimator, TransformerMixin):
	"""Raise numeric data to power."""

	def __init__(self, power):
		if not isinstance(power, int):
			raise TypeError("Power {0} is not an integer".format(power))
		self.power = power

	def fit(self, X, y = None):
		return self

	def transform(self, X):
		return numpy.power(X, self.power)

class ConcatTransformer(BaseEstimator, TransformerMixin):
	"""Concat data to string."""

	def __init__(self, separator = ""):
		self.separator = separator

	def fit(self, X, y = None):
		return self

	def transform(self, X):
		func = lambda x: self.separator.join([str(v) for v in x])
		Xt = eval_rows(X, func)
		return _col2d(Xt)

class MatchesTransformer(BaseEstimator, TransformerMixin):
	"""Match RE pattern."""

	def __init__(self, pattern):
		self.pattern = pattern

	def fit(self, X, y = None):
		X = ensure_1d(X)
		return self

	def transform(self, X):
		X = ensure_1d(X)
		engine = _regex_engine(self.pattern)
		func = lambda x: bool(engine.search(x))
		Xt = eval_rows(X, func)
		return _col2d(Xt)

class ReplaceTransformer(BaseEstimator, TransformerMixin):
	"""Replace all RE pattern matches."""

	def __init__(self, pattern, replacement):
		self.pattern = pattern
		self.replacement = replacement

	def fit(self, X, y = None):
		X = ensure_1d(X)
		return self

	def transform(self, X):
		X = ensure_1d(X)
		engine = _regex_engine(self.pattern)
		func = lambda x: engine.sub(self.replacement, x)
		Xt = eval_rows(X, func)
		return _col2d(Xt)

class SubstringTransformer(BaseEstimator, TransformerMixin):
	"""Extract substring."""

	def __init__(self, begin, end):
		if begin < 0:
			raise ValueError("Begin position {0} is negative".format(begin))
		if end < begin:
			raise ValueError("End position {0} is smaller than begin position {1}".format(end, begin))
		self.begin = begin
		self.end = end

	def fit(self, X, y = None):
		X = ensure_1d(X)
		return self

	def transform(self, X):
		X = ensure_1d(X)
		func = lambda x: x[self.begin:self.end]
		Xt = eval_rows(X, func)
		return _col2d(Xt)

class WordCountTransformer(BaseEstimator, TransformerMixin):
	"""Count tokens."""

	def __init__(self, word_pattern = "\w+", non_word_pattern = "\W+"):
		self.word_pattern = word_pattern
		self.non_word_pattern = non_word_pattern
		self.pipeline_ = Pipeline([
			("word_replacer", ReplaceTransformer(pattern = "({0})".format(word_pattern), replacement = "1")),
			("non_word_replacer", ReplaceTransformer(pattern = "({0})".format(non_word_pattern), replacement = "")),
			("counter", ExpressionTransformer("len(X[0])", dtype = int))
		])

	def fit(self, X, y = None):
		X = ensure_1d(X)
		return self

	def transform(self, X):
		X = ensure_1d(X)
		return self.pipeline_.transform(X)

class StringNormalizer(BaseEstimator, TransformerMixin):
	"""Normalize the case and surrounding whitespace."""

	def __init__(self, function = None, trim_blanks = True):
		functions = ["lower", "lowercase", "upper", "uppercase"]
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
		if self.function is None:
			pass
		elif self.function == "lower" or self.function == "lowercase":
			Xt = numpy.char.lower(Xt)
		elif self.function == "upper" or self.function == "uppercase":
			Xt = numpy.char.upper(Xt)
		else:
			raise ValueError(self.function)
		# Trim blanks
		if self.trim_blanks:
			Xt = numpy.char.strip(Xt)
		return Xt
