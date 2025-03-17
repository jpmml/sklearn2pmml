from collections import defaultdict
try:
	from collections.abc import Hashable
except ImportError:
	from collections import Hashable
from datetime import datetime
from io import StringIO
from pandas import CategoricalDtype, DataFrame, Series
from scipy.interpolate import BSpline
from scipy.sparse import lil_matrix
from sklearn.base import BaseEstimator, TransformerMixin
try:
	# SkLearn 1.2.0+
	from sklearn.base import OneToOneFeatureMixin
except ImportError:
	from sklearn.base import _OneToOneFeatureMixin as OneToOneFeatureMixin
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn2pmml import _is_pandas_categorical, _is_proto_pandas_categorical
from sklearn2pmml.preprocessing.regex import make_regex_engine
from sklearn2pmml.util import cast, check_expression, check_predicate, dt_transform, ensure_def, eval_rows, is_1d, to_1d, to_expr_func, to_numpy, Reshaper

import numpy
import pandas
import types

def _unique(X):
	nonmissing_mask = pandas.notnull(X)
	return numpy.unique(X[nonmissing_mask])

def _aggregate_fun(function):
	if function == "avg" or function == "mean":
		return numpy.nanmean
	elif function == "max":
		return numpy.nanmax
	elif function == "min":
		return numpy.nanmin
	elif function == "product" or function == "prod":
		return numpy.nanprod
	elif function == "sum":
		return numpy.nansum
	else:
		raise ValueError(function)

class AggregateTransformer(BaseEstimator, TransformerMixin):
	"""Aggregate continuous data."""

	def __init__(self, function):
		functions = ["avg", "min", "max", "mean", "prod", "product", "sum"]
		if function not in functions:
			raise ValueError("Function {0} not in {1}".format(function, functions))
		self.function = function

	def fit(self, X, y = None):
		return self

	def transform(self, X):
		fun = _aggregate_fun(self.function)
		Xt = fun(X, axis = 1)
		return Xt.reshape((-1, 1))

class BSplineTransformer(BaseEstimator, TransformerMixin):

	def __init__(self, bspline):
		if not isinstance(bspline, BSpline):
			raise TypeError("The spline object is not an instance of {0}".format(BSpline.__name__))
		self.bspline = bspline

	def fit(self, X, y = None):
		to_1d(X)
		return self

	def transform(self, X):
		X1d = to_1d(X)
		Xt = self.bspline(X1d)
		return Xt.reshape(X.shape)

class CastTransformer(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):
	"""Change data type."""

	def __init__(self, dtype):
		if isinstance(dtype, str) and dtype.startswith("datetime64"):
			dtypes = ["datetime64[D]", "datetime64[s]"]
			if dtype not in dtypes:
				raise ValueError("Temporal data type {0} not in {1}".format(dtype, dtypes))
		self.dtype = dtype

	def fit(self, X, y = None):
		if _is_proto_pandas_categorical(self.dtype):
			X = to_numpy(X)
			self.dtype_ = CategoricalDtype(categories = _unique(X), ordered = False)
		else:
			self.dtype_ = self.dtype
		return self

	def transform(self, X):
		return cast(X, self.dtype_)

class CutTransformer(BaseEstimator, TransformerMixin):
	"""Bin continuous data to categorical."""

	def __init__(self, bins, right = True, labels = None, include_lowest = True):
		self.bins = bins
		self.right = right
		self.labels = labels
		self.include_lowest = include_lowest

	def fit(self, X, y = None):
		to_1d(X)
		return self

	def transform(self, X):
		X1d = to_1d(X)
		Xt = pandas.cut(X1d, bins = self.bins, right = self.right, labels = self.labels, include_lowest = self.include_lowest)
		if _is_pandas_categorical(Xt.dtype):
			Xt = to_numpy(Xt)
		return Xt.reshape(X.shape)

class DataFrameConstructor(BaseEstimator, TransformerMixin):

	def __init__(self, columns, dtype):
		self.columns = columns
		self.dtype = dtype

	def get_feature_names_out(self, input_features = None):
		return numpy.asarray(self.columns)

	def fit(self, X, y = None):
		return self

	def transform(self, X):
		return DataFrame(X, columns = self.columns, dtype = self.dtype)

class SeriesConstructor(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):

	def __init__(self, name, dtype):
		self.name = name
		self.dtype = dtype

	def get_feature_names_out(self, input_features = None):
		if self.name is not None:
			return numpy.asarray([self.name])
		return super(SeriesConstructor, self).get_feature_names_out(input_features = input_features)

	def fit(self, X, y = None):
		to_1d(X)
		return self

	def transform(self, X):
		X1d = to_1d(X)
		return Series(X1d, name = self.name, dtype = self.dtype)

def _int(X):
	if numpy.isscalar(X):
		return int(X)
	else:
		return cast(X, int)

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
		self.expr = check_expression(expr)
		self.map_missing_to = map_missing_to
		self.default_value = default_value
		invalid_value_treatments = ["return_invalid", "as_missing"]
		if (invalid_value_treatment is not None) and (invalid_value_treatment not in invalid_value_treatments):
			raise ValueError("Invalid value treatment {0} not in {1}".format(invalid_value_treatment, invalid_value_treatments))
		self.invalid_value_treatment = invalid_value_treatment
		self.dtype = dtype

	def _eval(self, X):
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
			Xt = eval_rows(X, _eval_row, to_numpy = (not is_1d(X)), shape = (-1, 1))
		return Xt

	def fit(self, X, y = None):
		if _is_proto_pandas_categorical(self.dtype):
			Xt = self._eval(X)
			Xt = cast(Xt, self.dtype)
			dtype = Xt.dtype
		else:
			dtype = self.dtype
		self.dtype_ = dtype
		return self

	def transform(self, X):
		Xt = self._eval(X)
		if hasattr(self, "dtype_"):
			dtype = self.dtype_
		else:
			if _is_proto_pandas_categorical(self.dtype):
				raise NotFittedError()
			dtype = self.dtype
		if dtype is not None:
			Xt = cast(Xt, dtype)
		return Xt

	def fit_transform(self, X, y = None):
		Xt = self._eval(X)
		if self.dtype is not None:
			Xt = cast(Xt, self.dtype)
			if _is_proto_pandas_categorical(self.dtype):
				dtype = Xt.dtype
			else:
				dtype = self.dtype
		else:
			dtype = None
		self.dtype_ = dtype
		return Xt

class IdentityTransformer(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):
	"""Passes data through as-is."""

	def __init__(self):
		pass

	def fit(self, X, y = None):
		return self

	def transform(self, X):
		return X

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
		to_1d(X)
		return self

	def transform(self, X):
		X1d = to_1d(X)
		func = lambda x: self._strftime(x)
		Xt = eval_rows(X1d, func, shape = X.shape)
		return Xt

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
		to_1d(X)
		return self

	def transform(self, X):
		X1d = to_1d(X)
		func = lambda x: self._printf(x)
		Xt = eval_rows(X1d, func, shape = X.shape)
		return Xt

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
			if pandas.isnull(k):
				raise ValueError("Key is a missing value")
			if k_type is None:
				k_type = type(k)
			else:
				if type(k) != k_type:
					raise TypeError("Key is not a {0}".format(k_type.__name__))
			if pandas.isnull(v):
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
		to_1d(X)
		return self

	def transform(self, X):
		X1d = to_1d(X)
		transform_dict = self._transform_dict()

		def _eval_row(x):
			if pandas.isnull(x):
				return x
			return transform_dict[x]

		Xt = eval_rows(X1d, _eval_row, shape = X.shape)
		return Xt

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
			if pandas.isnull(v):
				raise ValueError("Value is a missing value")
			if type(k) != type(v):
				raise TypeError("Key and Value type mismatch")

	def _transform_dict(self):
		return self.mapping

	def transform(self, X):
		X1d = to_1d(X)
		transform_dict = self._transform_dict()

		def _eval_row(x):
			if x not in transform_dict:
				return x
			return transform_dict[x]

		Xt = eval_rows(X1d, _eval_row, shape = X.shape)
		return Xt

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

		def _eval_row(x):
			if (pandas.isnull(x)).any():
				return None
			# See https://stackoverflow.com/a/3460747
			# See https://stackoverflow.com/a/3338368
			x = x if isinstance(x, Hashable) else tuple(numpy.squeeze(numpy.asarray(x)))
			return transform_dict[tuple(x)]

		Xt = eval_rows(X, _eval_row, to_numpy = True)
		return Xt.reshape((-1, 1))

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
		X1d = to_1d(X)
		self.classes_ = _unique(X1d)
		return self

	def transform(self, X):
		X1d = to_1d(X)
		mapping = _make_index(self.classes_)
		if self.sparse_output:
			Xt = lil_matrix((len(X1d), len(mapping)), dtype = int)
		else:
			Xt = numpy.zeros((len(X1d), len(mapping)), dtype = int)
		for i, v in enumerate(X1d):
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
		X1d = to_1d(X)
		self.classes_ = _unique(X1d)
		return self

	def transform(self, X):
		X1d = to_1d(X)
		mapping = _make_index(self.classes_)
		Xt = numpy.array([self.missing_values if pandas.isnull(v) else mapping.get(v, self.missing_values) for v in X1d])
		return Xt.reshape((-1, 1))

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

class LagTransformer(BaseEstimator, TransformerMixin):

	def __init__(self, n):
		if not isinstance(n, int):
			raise TypeError("Shift {} is not an integer".format(n))
		if n < 1:
			raise ValueError("Shift {} is not a positive integer".format(n))
		self.n = n

	def fit(self, X, y = None):
		return self

	def transform(self, X):
		if hasattr(X, "shift"):
			return X.shift(self.n)
		else:
			X = numpy.asarray(X)
			if len(X.shape) != 2:
				raise ValueError("Expected a 2D array, got {}D array".format(len(X.shape)))
			Xt = numpy.roll(X, shift = self.n, axis = 0)
			Xt[:self.n, :] = numpy.nan
			return Xt

class RollingAggregateTransformer(BaseEstimator, TransformerMixin):

	def __init__(self, function, n):
		functions = ["avg", "min", "max", "mean", "prod", "product", "sum"]
		if function not in functions:
			raise ValueError("Function {0} not in {1}".format(function, functions))
		self.function = function
		if not isinstance(n, int):
			raise TypeError("Window size {} is not an integer".format(n))
		if n < 1:
			raise ValueError("Window size {} is not a positive integer".format(n))
		self.n = n

	def fit(self, X, y = None):
		return self

	def transform(self, X):
		fun = _aggregate_fun(self.function)
		if hasattr(X, "rolling"):
			X_windows = X.rolling(window = self.n, min_periods = 1, closed = "left")
			return X_windows.apply(fun, raw = True)
		else:
			X = numpy.asarray(X)
			if len(X.shape) != 2:
				raise ValueError("Expected a 2D array, got {}D array".format(len(X.shape)))
			Xt = numpy.full_like(X, fill_value = numpy.nan, dtype = float)
			for i in range(0, X.shape[0]):
				X_window = X[max(0, i - self.n):i]
				if X_window.size == 0:
					continue
				Xt[i] = fun(X_window, axis = 0)
			return Xt

class ConcatTransformer(BaseEstimator, TransformerMixin):
	"""Concat data to string."""

	def __init__(self, separator = ""):
		self.separator = separator

	def fit(self, X, y = None):
		return self

	def transform(self, X):
		func = lambda x: self.separator.join([str(v) for v in x])
		Xt = eval_rows(X, func, to_numpy = True)
		return Xt.reshape((-1, 1))

class RegExTransformer(BaseEstimator, TransformerMixin):

	def __init__(self, pattern, re_flavour):
		super(RegExTransformer, self).__init__()
		self.pattern = pattern
		re_flavours = ["pcre", "pcre2", "re"]
		if re_flavour not in re_flavours:
			raise ValueError("Regular Expressions flavour {0} not in {1}".format(re_flavour, re_flavours))
		self.re_flavour = re_flavour

	def _regex_func(self, regex_engine):
		raise NotImplementedError()

	def fit(self, X, y = None):
		to_1d(X)
		return self

	def transform(self, X):
		X1d = to_1d(X)
		regex_engine = make_regex_engine(self.pattern, self.re_flavour)
		func = self._regex_func(regex_engine)
		Xt = eval_rows(X1d, func, shape = X.shape)
		return Xt

class MatchesTransformer(RegExTransformer):
	"""Match RE pattern."""

	def __init__(self, pattern, re_flavour = "re"):
		super(MatchesTransformer, self).__init__(pattern = pattern, re_flavour = re_flavour)

	def _regex_func(self, regex_engine):
		def matches(x):
			return bool(regex_engine.matches(x))

		return matches

class ReplaceTransformer(RegExTransformer):
	"""Replace all RE pattern matches."""

	def __init__(self, pattern, replacement, re_flavour = "re"):
		super(ReplaceTransformer, self).__init__(pattern = pattern, re_flavour = re_flavour)
		self.replacement = replacement

	def _regex_func(self, regex_engine):
		def replace(x):
			return regex_engine.replace(self.replacement, x)

		return replace

class StringTransformer(BaseEstimator, TransformerMixin):

	def __init__(self):
		pass

	def fit(self, X, y = None):
		to_1d(X)
		return self

class StringLengthTransformer(StringTransformer):

	def __init__(self):
		super(StringLengthTransformer, self).__init__()

	def transform(self, X):
		X1d = to_1d(X)
		Xt = cast(X1d, "U")
		if hasattr(Xt, "str"):
			Xt = Xt.str.len()
			return Xt
		else:
			Xt = numpy.char.str_len(Xt)
			return Xt.reshape(X.shape)

class StringNormalizer(StringTransformer):
	"""Normalize the case and surrounding whitespace."""

	def __init__(self, function = None, trim_blanks = True):
		super(StringNormalizer, self).__init__()
		functions = ["lower", "lowercase", "upper", "uppercase"]
		if (function is not None) and (function not in functions):
			raise ValueError("Function {0} not in {1}".format(function, functions))
		self.function = function
		self.trim_blanks = trim_blanks

	def transform(self, X):
		X1d = to_1d(X)
		# Convert from any data type to Unicode string data type
		Xt = cast(X1d, "U")
		# Transform
		if self.function is None:
			pass
		elif self.function == "lower" or self.function == "lowercase":
			if hasattr(Xt, "str"):
				Xt = Xt.str.lower()
			else:
				Xt = numpy.char.lower(Xt)
		elif self.function == "upper" or self.function == "uppercase":
			if hasattr(Xt, "str"):
				Xt = Xt.str.upper()
			else:
				Xt = numpy.char.upper(Xt)
		else:
			raise ValueError(self.function)
		# Trim blanks
		if self.trim_blanks:
			if hasattr(Xt, "str"):
				Xt = Xt.str.strip()
			else:
				Xt = numpy.char.strip(Xt)
		if hasattr(Xt, "str"):
			return Xt
		else:
			return Xt.reshape(X.shape)

class SubstringTransformer(StringTransformer):
	"""Extract substring."""

	def __init__(self, begin, end):
		super(SubstringTransformer, self).__init__()
		if begin < 0:
			raise ValueError("Begin position {0} is negative".format(begin))
		if end < begin:
			raise ValueError("End position {0} is smaller than begin position {1}".format(end, begin))
		self.begin = begin
		self.end = end

	def transform(self, X):
		X1d = to_1d(X)
		# Convert from any data type to Unicode string data type
		Xt = cast(X1d, "U")
		if hasattr(Xt, "str"):
			Xt = Xt.str.slice(self.begin, self.end)
		else:
			func = lambda x: x[self.begin:self.end]
			Xt = eval_rows(Xt, func, shape = X.shape)
		return Xt

class WordCountTransformer(StringTransformer):
	"""Count tokens."""

	def __init__(self, word_pattern = r"\w+", non_word_pattern = r"\W+"):
		super(WordCountTransformer, self).__init__()
		self.word_pattern = word_pattern
		self.non_word_pattern = non_word_pattern
		self.pipeline_ = Pipeline([
			("word_replacer", ReplaceTransformer(pattern = "({0})".format(word_pattern), replacement = "1")),
			("non_word_replacer", ReplaceTransformer(pattern = "({0})".format(non_word_pattern), replacement = "")),
			("counter", StringLengthTransformer())
		])

	def transform(self, X):
		X1d = to_1d(X)
		# The expression "X]0]" assumes a two-dimensional array
		X1d = to_numpy(X1d).reshape((-1, 1))
		return self.pipeline_.transform(X1d)

def _to_sparse(X, step_mask, step_result):
	# Make array
	result = numpy.empty((X.shape[0], step_result.shape[1]), dtype = object)
	# Fill array
	result[step_mask.ravel(), :] = step_result
	return result

class SelectFirstTransformer(BaseEstimator, TransformerMixin):

	def __init__(self, steps, controller = None, eval_rows = True):
		for step in steps:
			if type(step) is not tuple:
				raise TypeError("Step is not a tuple")
			if len(step) != 3:
				raise TypeError("Step is not a three-element (name, transformer, predicate) tuple")
			name, transformer, predicate = step
			check_predicate(predicate)
		self.steps = steps
		if controller:
			if not hasattr(controller, "transform"):
				raise TypeError()
		self.controller = controller
		self.eval_rows = eval_rows

	def _to_evaluation_dataset(self, X):
		if self.controller is not None:
			return self.controller.transform(X)
		return X

	def _eval_step_mask(self, X, predicate):
		step_func = to_expr_func(predicate)
		if self.eval_rows:
			return eval_rows(X, step_func, dtype = bool)
		else:
			return step_func(X)

	def fit(self, X, y = None):
		X_eval = self._to_evaluation_dataset(X)
		mask = numpy.full(X.shape[0], fill_value = False)
		for name, transformer, predicate in self.steps:
			step_mask = numpy.logical_not(mask)
			step_mask_eval = self._eval_step_mask(X_eval[step_mask], predicate)
			if numpy.sum(step_mask_eval) < 1:
				raise ValueError(predicate)
			step_mask[step_mask] = step_mask_eval
			step_X = X[step_mask]
			step_y = y[step_mask] if y is not None else None
			transformer.fit(step_X, step_y)
			mask = numpy.logical_or(mask, step_mask)
		return self

	def transform(self, X):
		result = None
		X_eval = self._to_evaluation_dataset(X)
		mask = numpy.full(X.shape[0], fill_value = False)
		for name, transformer, predicate in self.steps:
			step_mask = numpy.logical_not(mask)
			step_mask_eval = self._eval_step_mask(X_eval[step_mask], predicate)
			if numpy.sum(step_mask_eval) < 1:
				continue
			step_mask[step_mask] = step_mask_eval
			step_X = X[step_mask]
			step_result = transformer.transform(step_X)
			step_result = _to_sparse(X, step_mask, step_result)
			if result is None:
				result = step_result
			else:
				result[step_mask] = step_result[step_mask]
			mask = numpy.logical_or(mask, step_mask)
		return result
