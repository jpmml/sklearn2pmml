from pandas import DataFrame
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.utils import column_or_1d
from sklearn2pmml.util import cast, eval_rows

import numpy
import pandas

class Alias(BaseEstimator, TransformerMixin):

	def __init__(self, transformer, name, prefit = False):
		self.transformer = transformer
		self.name = name
		self.prefit = prefit
		if prefit:
			self.transformer_ = clone(self.transformer)

	def fit(self, X, y = None):
		self.transformer_ = clone(self.transformer)
		if y is None:
			self.transformer_.fit(X)
		else:
			self.transformer_.fit(X, y)
		return self

	def transform(self, X):
		return self.transformer_.transform(X)

def _count(mask):
	if hasattr(mask, "values"):
		mask = mask.values
	missing_freq = sum(mask)
	non_missing_freq = sum(~mask)
	return {
		"totalFreq" : (missing_freq + non_missing_freq),
		"missingFreq" : missing_freq,
		"invalidFreq" : (non_missing_freq - non_missing_freq) # A scalar zero, or an array of zeroes
	}

class Domain(BaseEstimator, TransformerMixin):

	def __init__(self, missing_values = None, missing_value_treatment = "as_is", missing_value_replacement = None, invalid_value_treatment = "return_invalid", invalid_value_replacement = None, with_data = True, with_statistics = True, dtype = None, display_name = None):
		self.missing_values = missing_values
		missing_value_treatments = ["as_is", "as_mean", "as_mode", "as_median", "as_value", "return_invalid"]
		if missing_value_treatment not in missing_value_treatments:
			raise ValueError("Missing value treatment {0} not in {1}".format(missing_value_treatment, missing_value_treatments))
		self.missing_value_treatment = missing_value_treatment
		if missing_value_replacement is not None:
			if missing_value_treatment == "return_invalid":
				raise ValueError("Missing value treatment {0} does not support missing_value_replacement attribute", missing_value_treatment)
		self.missing_value_replacement = missing_value_replacement
		invalid_value_treatments = ["return_invalid", "as_is", "as_missing"]
		if invalid_value_treatment not in invalid_value_treatments:
			raise ValueError("Invalid value treatment {0} not in {1}".format(invalid_value_treatment, invalid_value_treatments))
		self.invalid_value_treatment = invalid_value_treatment
		if invalid_value_replacement is not None:
			if invalid_value_treatment == "return_invalid" or invalid_value_treatment == "as_missing":
				raise ValueError("Invalid value treatment {0} does not support invalid_value_replacement attribute", invalid_value_treatment)
		self.invalid_value_replacement = invalid_value_replacement
		self.with_data = with_data
		self.with_statistics = with_statistics
		self.dtype = dtype
		self.display_name = display_name

	def _empty_fit(self):
		return not (self.with_data or self.with_statistics)

	def _missing_value_mask(self, X):
		if self.missing_values is not None:
			def is_missing(X, missing_value):
				# float("NaN") != float("NaN")
				if isinstance(missing_value, float) and numpy.isnan(missing_value):
					return pandas.isnull(X)
				return X == missing_value
			if type(self.missing_values) is list:
				mask = numpy.full(X.shape, False, dtype = bool)
				for missing_value in self.missing_values:
					mask = numpy.logical_or(mask, is_missing(X, missing_value))
				return mask
			else:
				return is_missing(X, self.missing_values)
		else:
			return pandas.isnull(X)

	def _valid_value_mask(self, X, where):
		mask = numpy.full(X.shape, True, dtype = bool)
		return numpy.logical_and(mask, where)

	def _transform_missing_values(self, X, where):
		if self.missing_value_treatment == "return_invalid":
			if numpy.any(where) > 0:
				raise ValueError("Data contains {0} missing values".format(numpy.count_nonzero(where)))
		if self.missing_value_replacement is not None:
			X[where] = self.missing_value_replacement

	def _transform_valid_values(self, X, where):
		pass

	def _transform_invalid_values(self, X, where):
		if self.invalid_value_treatment == "return_invalid":
			if numpy.any(where) > 0:
				raise ValueError("Data contains {0} invalid values".format(numpy.count_nonzero(where)))
		elif self.invalid_value_treatment == "as_is":
			if self.invalid_value_replacement is not None:
				X[where] = self.invalid_value_replacement
		elif self.invalid_value_treatment == "as_missing":
			self._transform_missing_values(X, where)

	def transform(self, X):
		if self.dtype is not None:
			X = cast(X, self.dtype)
		missing_value_mask = self._missing_value_mask(X)
		nonmissing_value_mask = ~missing_value_mask
		valid_value_mask = self._valid_value_mask(X, nonmissing_value_mask)
		invalid_value_mask = ~numpy.logical_or(missing_value_mask, valid_value_mask)
		self._transform_missing_values(X, missing_value_mask)
		self._transform_valid_values(X, valid_value_mask)
		self._transform_invalid_values(X, invalid_value_mask)
		return X

class DiscreteDomain(Domain):

	def __init__(self, missing_values = None, missing_value_treatment = "as_is", missing_value_replacement = None, invalid_value_treatment = "return_invalid", invalid_value_replacement = None, with_data = True, with_statistics = True, dtype = None, display_name = None):
		super(DiscreteDomain, self).__init__(missing_values = missing_values, missing_value_treatment = missing_value_treatment, missing_value_replacement = missing_value_replacement, invalid_value_treatment = invalid_value_treatment, invalid_value_replacement = invalid_value_replacement, with_data = with_data, with_statistics = with_statistics, dtype = dtype, display_name = display_name)

	def _valid_value_mask(self, X, where):
		if hasattr(self, "data_"):
			if hasattr(X, "isin"):
				mask = X.isin(self.data_)
			else:
				def is_valid(x):
					return x in self.data_
				mask = eval_rows(X, is_valid, dtype = bool)
			mask = (numpy.asarray(mask, dtype = bool)).reshape(X.shape)
			return numpy.logical_and(mask, where)
		return super(DiscreteDomain, self)._valid_value_mask(X, where)

	def fit(self, X, y = None):
		X = column_or_1d(X, warn = True)
		if self._empty_fit():
			return self
		if self.dtype is not None:
			X = cast(X, self.dtype)
		mask = self._missing_value_mask(X)
		values, counts = numpy.unique(X[~mask], return_counts = True)
		if self.with_data:
			if (self.missing_value_replacement is not None) and numpy.any(mask) > 0:
				self.data_ = numpy.unique(numpy.append(values, self.missing_value_replacement))
			else:
				self.data_ = values
		if self.with_statistics:
			self.counts_ = _count(mask)
			self.discr_stats_ = (values, counts)
		return self

class CategoricalDomain(DiscreteDomain):

	def __init__(self, missing_values = None, missing_value_treatment = "as_is", missing_value_replacement = None, invalid_value_treatment = "return_invalid", invalid_value_replacement = None, with_data = True, with_statistics = True, dtype = None, display_name = None):
		super(CategoricalDomain, self).__init__(missing_values = missing_values, missing_value_treatment = missing_value_treatment, missing_value_replacement = missing_value_replacement, invalid_value_treatment = invalid_value_treatment, invalid_value_replacement = invalid_value_replacement, with_data = with_data, with_statistics = with_statistics, dtype = dtype, display_name = display_name)

class OrdinalDomain(DiscreteDomain):

	def __init__(self, missing_values = None, missing_value_treatment = "as_is", missing_value_replacement = None, invalid_value_treatment = "return_invalid", invalid_value_replacement = None, with_data = True, with_statistics = True, dtype = None, display_name = None):
		super(OrdinalDomain, self).__init__(missing_values = missing_values, missing_value_treatment = missing_value_treatment, missing_value_replacement = missing_value_replacement, invalid_value_treatment = invalid_value_treatment, invalid_value_replacement = invalid_value_replacement, with_data = with_data, with_statistics = with_statistics, dtype = dtype, display_name = display_name)

def _interquartile_range(X, axis):
	quartiles = numpy.nanpercentile(X, [25, 75], axis = axis)
	return (quartiles[1] - quartiles[0])

class ContinuousDomain(Domain):

	def __init__(self, missing_values = None, missing_value_treatment = "as_is", missing_value_replacement = None, invalid_value_treatment = "return_invalid", invalid_value_replacement = None, outlier_treatment = "as_is", low_value = None, high_value = None, with_data = True, with_statistics = True, dtype = None, display_name = None):
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

	def _valid_value_mask(self, X, where):
		if hasattr(self, "data_min_") and hasattr(self, "data_max_"):
			mask = (numpy.greater_equal(X, self.data_min_, where = where) & numpy.less_equal(X, self.data_max_, where = where))
			return numpy.logical_and(mask, where)
		return super(ContinuousDomain, self)._valid_value_mask(X, where)

	def fit(self, X, y = None):
		if self._empty_fit():
			return self
		if self.dtype is not None:
			X = cast(X, self.dtype)
		mask = self._missing_value_mask(X)
		X = numpy.ma.masked_array(X, mask = mask)
		min = numpy.asarray(numpy.nanmin(X, axis = 0))
		max = numpy.asarray(numpy.nanmax(X, axis = 0))
		if self.with_data:
			self.data_min_ = min
			self.data_max_ = max
		if self.with_statistics:
			self.counts_ = _count(mask)
			X = numpy.ma.asarray(X, dtype = numpy.float).filled(float("NaN"))
			self.numeric_info_ = {
				"minimum" : min,
				"maximum" : max,
				"mean" : numpy.asarray(numpy.nanmean(X, axis = 0)),
				"standardDeviation" : numpy.asarray(numpy.nanstd(X, axis = 0)),
				"median" : numpy.asarray(numpy.nanmedian(X, axis = 0)),
				"interQuartileRange" : numpy.asarray(_interquartile_range(X, axis = 0))
			}
		return self

	def _outlier_mask(self, X, where):
		mask = (numpy.less(X, self.low_value, where = where) | numpy.greater(X, self.high_value, where = where))
		return numpy.logical_and(mask, where)

	def _negative_outlier_mask(self, X, where):
		mask = numpy.less(X, self.low_value, where = where)
		return numpy.logical_and(mask, where)

	def _positive_outlier_mask(self, X, where):
		mask = numpy.greater(X, self.high_value, where = where)
		return numpy.logical_and(mask, where)

	def _transform_valid_values(self, X, where):
		if self.outlier_treatment == "as_missing_values":
			mask = self._outlier_mask(X, where)
			if self.missing_values is not None:
				if type(self.missing_values) is list:
					raise ValueError()
				X[mask] = self.missing_values
			else:
				X[mask] = None
			self._transform_missing_values(X, mask)
		elif self.outlier_treatment == "as_extreme_values":
			mask = self._negative_outlier_mask(X, where)
			X[mask] = self.low_value
			mask = self._positive_outlier_mask(X, where)
			X[mask] = self.high_value

class TemporalDomain(Domain):

	def __init__(self, missing_values = None, missing_value_treatment = "as_is", missing_value_replacement = None, invalid_value_treatment = "return_invalid", invalid_value_replacement = None, dtype = None, display_name = None):
		super(TemporalDomain, self).__init__(missing_values = missing_values, missing_value_treatment = missing_value_treatment, missing_value_replacement = missing_value_replacement, invalid_value_treatment = invalid_value_treatment, invalid_value_replacement = invalid_value_replacement, with_data = False, with_statistics = False, dtype = dtype, display_name = display_name)
		dtypes = ["datetime64[D]", "datetime64[s]"]
		if (not isinstance(dtype, str)) or (dtype not in dtypes):
			raise ValueError("Temporal data type {0} not in {1}".format(dtype, dtypes))

	def fit(self, X, y = None):
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
		rows, columns = X.shape
		if len(self.domains) != columns:
			raise ValueError("The number of columns {0} is not equal to the number of domain objects {1}".format(columns, len(self.domains)))
		if isinstance(X, DataFrame):
			for domain, column in zip(self.domains, X.columns):
				if domain is not None:
					domain.fit(X[column].values)
		else:
			for domain, column in zip(self.domains, range(0, columns)):
				if domain is not None:
					domain.fit(X[:, column])
		return self

	def transform(self, X):
		rows, columns = X.shape
		if len(self.domains) != columns:
			raise ValueError("The number of columns {0} is not equal to the number of domain objects {1}".format(columns, len(self.domains)))
		if isinstance(X, DataFrame):
			for domain, column in zip(self.domains, X.columns):
				if domain is not None:
					X[column] = domain.transform(X[column].values)
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