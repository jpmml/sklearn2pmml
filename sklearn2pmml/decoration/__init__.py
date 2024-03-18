from pandas import CategoricalDtype, DataFrame
from pandas.api.types import is_object_dtype
from sklearn.base import clone, BaseEstimator, TransformerMixin
try:
	# SkLearn 1.2.0+
	from sklearn.base import OneToOneFeatureMixin
except ImportError:
	from sklearn.base import _OneToOneFeatureMixin as OneToOneFeatureMixin
from sklearn2pmml import _is_pandas_categorical, _is_proto_pandas_categorical
from sklearn2pmml.util import cast, common_dtype, is_1d, to_numpy

import copy
import itertools
import numpy
import numbers
import pandas

class TransformerWrapper(BaseEstimator, TransformerMixin):

	def __init__(self, transformer, prefit = False):
		self.transformer = transformer
		self.prefit = prefit
		if prefit:
			self.transformer_ = copy.deepcopy(self.transformer)

	def fit(self, X, y = None, **fit_params):
		self.transformer_ = clone(self.transformer)
		if y is None:
			self.transformer_.fit(X, **fit_params)
		else:
			self.transformer_.fit(X, y, **fit_params)
		return self

	def transform(self, X):
		return self.transformer_.transform(X)

class Alias(TransformerWrapper):

	def __init__(self, transformer, name, prefit = False):
		super(Alias, self).__init__(transformer = transformer, prefit = prefit)
		if not isinstance(name, str):
			raise TypeError("Name is not a string")
		self.name = name

	def get_feature_names(self, input_features = None):
		return self.get_feature_names_out(input_features)

	def get_feature_names_out(self, input_features = None):
		return numpy.asarray([self.name])

class MultiAlias(TransformerWrapper):

	def __init__(self, transformer, names, prefit = True):
		super(MultiAlias, self).__init__(transformer = transformer, prefit = prefit)
		for name in names:
			if not isinstance(name, str):
				raise TypeError("Name is not a string")
		self.names = names

	def get_feature_names(self, input_features = None):
		return self.get_feature_names_out(input_features)

	def get_feature_names_out(self, input_features = None):
		return numpy.asarray(self.names)

def _check_input(estimator, X, reset):
	estimator._check_n_features(X, reset = reset)
	estimator._check_feature_names(X, reset = reset)
	return X

def _check_cols(X, values):
	if is_1d(X):
		if hasattr(values, "__len__") and len(values) > 1:
			raise ValueError()
	else:
		if X.shape[1] != len(values):
			raise ValueError()

def _set_values(X, where, values):
	X[where] = values
	return X

def _count(missing_mask, valid_mask, invalid_mask):
	missing_freq = sum(missing_mask)
	valid_freq = sum(valid_mask)
	invalid_freq = sum(invalid_mask) if (invalid_mask is not None) else (valid_freq - valid_freq) # A scalar zero, or an array of zeroes
	return {
		"totalFreq" : (missing_freq + valid_freq + invalid_freq),
		"missingFreq" : missing_freq,
		"invalidFreq" : invalid_freq
	}

class Domain(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):

	def __init__(self, missing_values = None, missing_value_treatment = "as_is", missing_value_replacement = None, invalid_value_treatment = "return_invalid", invalid_value_replacement = None, with_data = True, with_statistics = False, dtype = None, display_name = None):
		self.missing_values = missing_values
		missing_value_treatments = ["as_is", "as_mean", "as_mode", "as_median", "as_value", "return_invalid"]
		if missing_value_treatment not in missing_value_treatments:
			raise ValueError("Missing value treatment {0} not in {1}".format(missing_value_treatment, missing_value_treatments))
		self.missing_value_treatment = missing_value_treatment
		if missing_value_replacement is not None:
			if missing_value_treatment == "return_invalid":
				raise ValueError("Missing value treatment {0} does not support missing_value_replacement attribute".format(missing_value_treatment))
		self.missing_value_replacement = missing_value_replacement
		invalid_value_treatments = ["return_invalid", "as_is", "as_missing", "as_value"]
		if invalid_value_treatment not in invalid_value_treatments:
			raise ValueError("Invalid value treatment {0} not in {1}".format(invalid_value_treatment, invalid_value_treatments))
		self.invalid_value_treatment = invalid_value_treatment
		if invalid_value_replacement is not None:
			if invalid_value_treatment != "as_value":
				raise ValueError("Invalid value treatment {0} does not support invalid_value_replacement attribute".format(invalid_value_treatment))
		else:
			if invalid_value_treatment == "as_value":
				raise ValueError("Invalid value treatment {0} requires invalid_value_replacement attribute".format(invalid_value_treatment))
		self.invalid_value_replacement = invalid_value_replacement
		self.with_data = with_data
		self.with_statistics = with_statistics
		self.dtype = dtype
		self.display_name = display_name

	def _empty_fit(self):
		return not (self.with_data or self.with_statistics)

	def _to_missing(self, X, where):
		if not numpy.any(where):
			return X
		missing_value = None
		if self.missing_values is not None:
			if type(self.missing_values) is list:
				missing_value = self.missing_values[0]
			else:
				missing_value = self.missing_values
		X = _set_values(X, where, missing_value)
		return X

	def _missing_value_mask(self, X):
		if self.missing_values is not None:
			def is_missing(X, missing_value):
				# Values like float("NaN"), Numpy.NaN and Pandas.NA fail the '==' operator
				if pandas.isnull(missing_value):
					return pandas.isnull(X)
				return X == missing_value

			if type(self.missing_values) is list:
				mask = numpy.full(X.shape, fill_value = False)
				for missing_value in self.missing_values:
					mask = numpy.logical_or(mask, is_missing(X, missing_value))
				return mask
			else:
				return is_missing(X, self.missing_values)
		else:
			return pandas.isnull(X)

	def _valid_value_mask(self, X, where):
		return where

	def _transform_missing_values(self, X, where):
		if not numpy.any(where):
			return X
		if self.missing_value_treatment == "return_invalid":
			raise ValueError("Data contains {0} missing values".format(numpy.count_nonzero(where)))
		elif self.missing_value_treatment in ["as_is", "as_mean", "as_mode", "as_median", "as_value"]:
			if self.missing_value_replacement is not None:
				X = _set_values(X, where, self.missing_value_replacement)
			# Special case for object data type columns: replacing non-None values with None values
			elif is_object_dtype(self.dtype_) and (self.missing_value_replacement is None):
				X = _set_values(X, where, self.missing_value_replacement)
		else:
			raise ValueError()
		return X

	def _transform_valid_values(self, X, where):
		return X

	def _transform_invalid_values(self, X, where):
		if not numpy.any(where):
			return X
		if self.invalid_value_treatment == "return_invalid":
			raise ValueError("Data contains {0} invalid values".format(numpy.count_nonzero(where)))
		elif self.invalid_value_treatment == "as_is":
			pass
		elif self.invalid_value_treatment == "as_missing":
			X = self._to_missing(X, where)
			X = self._transform_missing_values(X, where)
		elif self.invalid_value_treatment == "as_value":
			if self.invalid_value_replacement is not None:
				X = _set_values(X, where, self.invalid_value_replacement)
		else:
			raise ValueError()
		return X

	def _compute_masks(self, X):
		X = to_numpy(X)
		missing_mask = self._missing_value_mask(X)
		nonmissing_mask = ~missing_mask
		valid_mask = self._valid_value_mask(X, nonmissing_mask)
		invalid_mask = ~numpy.logical_or(missing_mask, valid_mask)
		return (missing_mask, valid_mask, invalid_mask)

	def _should_make_copy(self, X, missing_mask, valid_mask, invalid_mask):
		if numpy.any(missing_mask) or numpy.any(invalid_mask):
			return True
		return False

	def transform(self, X):
		_check_input(self, X, reset = False)
		if self.dtype is not None:
			X = cast(X, self.dtype)
		missing_mask, valid_mask, invalid_mask = self._compute_masks(X)
		if self._should_make_copy(X, missing_mask, valid_mask, invalid_mask):
			X = X.copy()
		X = self._transform_missing_values(X, missing_mask)
		X = self._transform_valid_values(X, valid_mask)
		X = self._transform_invalid_values(X, invalid_mask)
		return X

class DiscreteDomain(Domain):

	def __init__(self, missing_values = None, missing_value_treatment = "as_is", missing_value_replacement = None, invalid_value_treatment = "return_invalid", invalid_value_replacement = None, with_data = True, with_statistics = False, dtype = None, display_name = None, data_values = None):
		super(DiscreteDomain, self).__init__(missing_values = missing_values, missing_value_treatment = missing_value_treatment, missing_value_replacement = missing_value_replacement, invalid_value_treatment = invalid_value_treatment, invalid_value_replacement = invalid_value_replacement, with_data = with_data, with_statistics = with_statistics, dtype = dtype, display_name = display_name)
		if data_values:
			if not with_data:
				raise ValueError("Valid values require with_data attribute")
			if isinstance(dtype, CategoricalDtype) and data_values != (dtype.categories).tolist():
				raise ValueError("Valid values are invalid")
		self.data_values = data_values

	def _is_ordered(self):
		raise NotImplementedError()

	def _valid_value_mask(self, X, where):
		if hasattr(self, "data_values_"):
			data_values = self.data_values_
		elif _is_pandas_categorical(self.dtype_):
			data_values = self.dtype_.categories
		else:
			data_values = None
		if data_values is not None:
			def _isin_mask(x, values):
				if hasattr(x, "isin"):
					return x.isin(values)
				else:
					return numpy.isin(x, values)

			if is_1d(X):
				where = where.ravel()
				mask = numpy.full(X.shape, fill_value = False)
				mask[where] = _isin_mask(X[where], data_values)
				return mask
			else:
				if hasattr(data_values, "__len__"):
					_check_cols(X, data_values)
				mask = numpy.full(X.shape, fill_value = False)
				for col in range(X.shape[1]):
					col_where = where[:, col]
					col_data_values = data_values[col] if hasattr(data_values, "__len__") else data_values
					mask[col_where, col] = _isin_mask(X[col_where, col], col_data_values)
				return mask
		return super(DiscreteDomain, self)._valid_value_mask(X, where)

	def fit(self, X, y = None):
		_check_input(self, X, reset = True)
		if self.dtype is not None:
			if _is_proto_pandas_categorical(self.dtype):
				if self.data_values is not None:
					dtype = CategoricalDtype(list(itertools.chain.from_iterable(self.data_values)), ordered = self._is_ordered())
				else:
					dtype = self.dtype
				X = cast(X, dtype)
			else:
				X = cast(X, self.dtype)
		self.dtype_ = common_dtype(X)
		if self._empty_fit():
			return self
		X = to_numpy(X)
		if self.with_data:
			if self.data_values is None:
				missing_mask = self._missing_value_mask(X)
				nonmissing_mask = ~missing_mask
			else:
				_check_cols(X, self.data_values)
			if is_1d(X):
				if self.data_values is None:
					if _is_pandas_categorical(self.dtype_):
						data_values = self.dtype_.categories
					else:
						data_values = numpy.unique(X[nonmissing_mask])
				else:
					data_values = numpy.asarray(self.data_values)
				self.data_values_ = data_values
			else:
				if self.data_values is None:
					if _is_pandas_categorical(self.dtype_):
						raise ValueError()
				self.data_values_ = []
				for col in range(X.shape[1]):
					if self.data_values is None:
						col_X = X[:, col]
						col_missing_mask = missing_mask[:, col]
						col_nonmissing_mask = nonmissing_mask[:, col]
						data_values = numpy.unique(col_X[col_nonmissing_mask])
					else:
						data_values = numpy.asarray(self.data_values[col])
					self.data_values_.append(data_values)
		if self.with_statistics:
			missing_mask, valid_mask, invalid_mask = self._compute_masks(X)
			self.counts_ = _count(missing_mask, valid_mask, invalid_mask)
			if is_1d(X):
				values, counts = numpy.unique(X[valid_mask], return_counts = True)
				self.discr_stats_ = (values, counts)
			else:
				self.discr_stats_ = []
				for col in range(X.shape[1]):
					col_X = X[:, col]
					col_valid_mask = valid_mask[:, col]
					values, counts = numpy.unique(col_X[col_valid_mask], return_counts = True)
					self.discr_stats_.append((values, counts))
		return self

class CategoricalDomain(DiscreteDomain):

	def __init__(self, missing_values = None, missing_value_treatment = "as_is", missing_value_replacement = None, invalid_value_treatment = "return_invalid", invalid_value_replacement = None, with_data = True, with_statistics = False, dtype = None, display_name = None, data_values = None):
		super(CategoricalDomain, self).__init__(missing_values = missing_values, missing_value_treatment = missing_value_treatment, missing_value_replacement = missing_value_replacement, invalid_value_treatment = invalid_value_treatment, invalid_value_replacement = invalid_value_replacement, with_data = with_data, with_statistics = with_statistics, dtype = dtype, display_name = display_name, data_values = data_values)
		if isinstance(dtype, CategoricalDtype) and dtype.ordered:
			raise ValueError()

	def _is_ordered(self):
		return False

class OrdinalDomain(DiscreteDomain):

	def __init__(self, missing_values = None, missing_value_treatment = "as_is", missing_value_replacement = None, invalid_value_treatment = "return_invalid", invalid_value_replacement = None, with_data = True, with_statistics = False, dtype = None, display_name = None, data_values = None):
		super(OrdinalDomain, self).__init__(missing_values = missing_values, missing_value_treatment = missing_value_treatment, missing_value_replacement = missing_value_replacement, invalid_value_treatment = invalid_value_treatment, invalid_value_replacement = invalid_value_replacement, with_data = with_data, with_statistics = with_statistics, dtype = dtype, display_name = display_name, data_values = data_values)
		if isinstance(dtype, CategoricalDtype) and not dtype.ordered:
			raise ValueError()

	def _is_ordered(self):
		return True

def _interquartile_range(X, axis):
	quartiles = numpy.nanpercentile(X, [25, 75], axis = axis)
	return (quartiles[1] - quartiles[0])

class ContinuousDomain(Domain):

	def __init__(self, missing_values = None, missing_value_treatment = "as_is", missing_value_replacement = None, invalid_value_treatment = "return_invalid", invalid_value_replacement = None, outlier_treatment = "as_is", low_value = None, high_value = None, with_data = True, with_statistics = False, dtype = None, display_name = None, data_min = None, data_max = None):
		super(ContinuousDomain, self).__init__(missing_values = missing_values, missing_value_treatment = missing_value_treatment, missing_value_replacement = missing_value_replacement, invalid_value_treatment = invalid_value_treatment, invalid_value_replacement = invalid_value_replacement, with_data = with_data, with_statistics = with_statistics, dtype = dtype, display_name = display_name)
		outlier_treatments = ["as_is", "as_missing_values", "as_extreme_values"]
		if outlier_treatment not in outlier_treatments:
			raise ValueError("Outlier treatment {0} not in {1}".format(outlier_treatment, outlier_treatments))
		self.outlier_treatment = outlier_treatment
		if outlier_treatment == "as_is":
			if (low_value is not None) or (high_value is not None):
				raise ValueError("Outlier treatment {0} does not support low_value and high_value attributes".format(outlier_treatment))
		elif outlier_treatment == "as_missing_values" or outlier_treatment == "as_extreme_values":
			if (low_value is None) or (high_value is None):
				raise ValueError("Outlier treatment {0} requires low_value and high_value attributes".format(outlier_treatment))
		self.low_value = low_value
		self.high_value = high_value
		if data_min or data_max:
			if not with_data:
				raise ValueError("Valid value intervals require with_data attribute")
		if data_min and data_max:
			if numpy.any(numpy.greater(data_min, data_max)):
				raise ValueError("Valid value intervals are invalid")
		self.data_min = data_min
		self.data_max = data_max

	def _valid_value_mask(self, X, where):
		if hasattr(self, "data_min_") and hasattr(self, "data_max_"):
			return numpy.where((X >= self.data_min_) & (X <= self.data_max_), where, False)
		return super(ContinuousDomain, self)._valid_value_mask(X, where)

	def fit(self, X, y = None):
		_check_input(self, X, reset = True)
		if self.dtype is not None:
			X = cast(X, self.dtype)
		self.dtype_ = common_dtype(X)
		if self._empty_fit():
			return self
		X = to_numpy(X)
		if self.with_data:
			dtype = self.dtype_
			# Unbox Pandas' extension data type to Numpy data type
			if hasattr(dtype, "numpy_dtype"):
				dtype = dtype.numpy_dtype
			if issubclass(dtype.type, numbers.Integral):
				info = numpy.iinfo(dtype)
			else:
				info = numpy.finfo(dtype)
			missing_mask = self._missing_value_mask(X)
			nonmissing_mask = ~missing_mask
			if self.data_min is None:
				self.data_min_ = numpy.asarray(numpy.nanmin(X, axis = 0, initial = info.max, where = nonmissing_mask))
			else:
				_check_cols(X, self.data_min)
				self.data_min_ = numpy.asarray(self.data_min)
			if self.data_max is None:
				self.data_max_ = numpy.asarray(numpy.nanmax(X, axis = 0, initial = info.min, where = nonmissing_mask))
			else:
				_check_cols(X, self.data_max)
				self.data_max_ = numpy.asarray(self.data_max)
		if self.with_statistics:
			missing_mask, valid_mask, invalid_mask = self._compute_masks(X)
			self.counts_ = _count(missing_mask, valid_mask, invalid_mask)
			if numpy.any(missing_mask) or numpy.any(invalid_mask):
				X = (X.copy()).astype(float)
				X[missing_mask | invalid_mask] = float("NaN")
			self.numeric_info_ = {
				"minimum" : numpy.asarray(numpy.nanmin(X, axis = 0)),
				"maximum" : numpy.asarray(numpy.nanmax(X, axis = 0)),
				"mean" : numpy.asarray(numpy.nanmean(X, axis = 0)),
				"standardDeviation" : numpy.asarray(numpy.nanstd(X, axis = 0)),
				"median" : numpy.asarray(numpy.nanmedian(X, axis = 0)),
				"interQuartileRange" : numpy.asarray(_interquartile_range(X, axis = 0))
			}
		return self

	def _outlier_mask(self, X, where):
		return numpy.where((X < self.low_value) | (X > self.high_value), where, False)

	def _negative_outlier_mask(self, X, where):
		return numpy.where(X < self.low_value, where, False)

	def _positive_outlier_mask(self, X, where):
		return numpy.where(X > self.high_value, where, False)

	def _transform_valid_values(self, X, where):
		if not numpy.any(where):
			return X
		if self.outlier_treatment == "as_missing_values":
			outlier_mask = self._outlier_mask(X, where)
			X = self._to_missing(X, outlier_mask)
			X = self._transform_missing_values(X, outlier_mask)
		elif self.outlier_treatment == "as_extreme_values":
			outlier_mask = self._negative_outlier_mask(X, where)
			X = _set_values(X, outlier_mask, self.low_value)
			outlier_mask = self._positive_outlier_mask(X, where)
			X = _set_values(X, outlier_mask, self.high_value)
		return X

	def _should_make_copy(self, X, missing_mask, valid_mask, invalid_mask):
		if super(ContinuousDomain, self)._should_make_copy(X, missing_mask, valid_mask, invalid_mask):
			return True
		if self.outlier_treatment == "as_missing_values":
			outlier_mask = self._outlier_mask(X, valid_mask)
			if numpy.any(outlier_mask):
				return True
		elif self.outlier_treatment == "as_extreme_values":
			outlier_mask = self._negative_outlier_mask(X, valid_mask)
			if numpy.any(outlier_mask):
				return True
			outlier_mask = self._positive_outlier_mask(X, valid_mask)
			if numpy.any(outlier_mask):
				return True
		return False

class TemporalDomain(Domain):

	def __init__(self, missing_values = None, missing_value_treatment = "as_is", missing_value_replacement = None, invalid_value_treatment = "return_invalid", invalid_value_replacement = None, dtype = None, display_name = None):
		super(TemporalDomain, self).__init__(missing_values = missing_values, missing_value_treatment = missing_value_treatment, missing_value_replacement = missing_value_replacement, invalid_value_treatment = invalid_value_treatment, invalid_value_replacement = invalid_value_replacement, with_data = False, with_statistics = False, dtype = dtype, display_name = display_name)
		dtypes = ["datetime64[D]", "datetime64[s]"]
		if (not isinstance(dtype, str)) or (dtype not in dtypes):
			raise ValueError("Temporal data type {0} not in {1}".format(dtype, dtypes))

	def fit(self, X, y = None):
		_check_input(self, X, reset = True)
		self.dtype_ = common_dtype(X)
		return self

class DateDomain(TemporalDomain):

	def __init__(self, missing_values = None, missing_value_treatment = "as_is", missing_value_replacement = None, invalid_value_treatment = "return_invalid", invalid_value_replacement = None, display_name = None):
		super(DateDomain, self).__init__(missing_values = missing_values, missing_value_treatment = missing_value_treatment, missing_value_replacement = missing_value_replacement, invalid_value_treatment = invalid_value_treatment, invalid_value_replacement = invalid_value_replacement, dtype = "datetime64[D]", display_name = display_name)

class DateTimeDomain(TemporalDomain):

	def __init__(self, missing_values = None, missing_value_treatment = "as_is", missing_value_replacement = None, invalid_value_treatment = "return_invalid", invalid_value_replacement = None, display_name = None):
		super(DateTimeDomain, self).__init__(missing_values = missing_values, missing_value_treatment = missing_value_treatment, missing_value_replacement = missing_value_replacement, invalid_value_treatment = invalid_value_treatment, invalid_value_replacement = invalid_value_replacement, dtype = "datetime64[s]", display_name = display_name)

class MultiDomain(BaseEstimator, TransformerMixin):

	def __init__(self, domains):
		self.domains = domains

	def fit(self, X, y = None):
		_check_input(self, X, reset = True)
		rows, columns = X.shape
		if len(self.domains) != columns:
			raise ValueError("The number of columns {0} is not equal to the number of domain objects {1}".format(columns, len(self.domains)))
		if isinstance(X, DataFrame):
			for domain, column in zip(self.domains, X.columns):
				if domain is not None:
					domain.fit(X[column])
		else:
			for domain, column in zip(self.domains, range(0, columns)):
				if domain is not None:
					domain.fit(X[:, column])
		return self

	def transform(self, X):
		_check_input(self, X, reset = False)
		rows, columns = X.shape
		# XXX
		X = X.copy()
		if isinstance(X, DataFrame):
			for domain, column in zip(self.domains, X.columns):
				if domain is not None:
					X[column] = domain.transform(X[column])
		else:
			for domain, column in zip(self.domains, range(0, columns)):
				if domain is not None:
					X[:, column] = domain.transform(X[:, column])
		return X

class DomainEraser(BaseEstimator, TransformerMixin):

	def __init__(self):
		pass

	def fit(self, X, y = None):
		return self

	def transform(self, X):
		return X

class ContinuousDomainEraser(DomainEraser):

	def __init__(self):
		super(ContinuousDomainEraser, self).__init__()

class DiscreteDomainEraser(DomainEraser):

	def __init__(self):
		super(DiscreteDomainEraser, self).__init__()