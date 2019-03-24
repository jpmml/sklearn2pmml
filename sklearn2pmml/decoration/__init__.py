from pandas import DataFrame
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.utils import column_or_1d

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

	def __init__(self, missing_values = None, missing_value_treatment = "as_is", missing_value_replacement = None, invalid_value_treatment = "return_invalid", invalid_value_replacement = None, with_data = True, with_statistics = True):
		if missing_values is not None:
			self.missing_values = missing_values
		missing_value_treatments = ["as_is", "as_mean", "as_mode", "as_median", "as_value", "return_invalid"]
		if missing_value_treatment not in missing_value_treatments:
			raise ValueError("Missing value treatment {0} not in {1}".format(missing_value_treatment, missing_value_treatments))
		self.missing_value_treatment = missing_value_treatment
		if missing_value_replacement is not None:
			self.missing_value_replacement = missing_value_replacement
		invalid_value_treatments = ["return_invalid", "as_is", "as_missing"]
		if invalid_value_treatment not in invalid_value_treatments:
			raise ValueError("Invalid value treatment {0} not in {1}".format(invalid_value_treatment, invalid_value_treatments))
		if invalid_value_replacement is not None:
			self.invalid_value_replacement = invalid_value_replacement
		self.invalid_value_treatment = invalid_value_treatment
		self.with_data = with_data
		self.with_statistics = with_statistics

	def _empty_fit(self):
		return not (self.with_data or self.with_statistics)

	def _missing_value_mask(self, X):
		if hasattr(self, "missing_values"):
			def is_missing(X, missing_value):
				# float("NaN") != float("NaN")
				if isinstance(missing_value, float) and numpy.isnan(missing_value):
					return pandas.isnull(X)
				return X == missing_value
			if type(self.missing_values) is list:
				mask = None
				for missing_value in self.missing_values:
					if mask is None:
						mask = is_missing(X, missing_value)
					else:
						mask = numpy.logical_or(mask, is_missing(X, missing_value))
				return mask
			else:
				return is_missing(X, self.missing_values)
		else:
			return pandas.isnull(X)

	def transform(self, X):
		if hasattr(self, "missing_value_replacement"):
			mask = self._missing_value_mask(X)
			X[mask] = self.missing_value_replacement
		return X

class CategoricalDomain(Domain):

	def __init__(self, missing_values = None, missing_value_treatment = "as_is", missing_value_replacement = None, invalid_value_treatment = "return_invalid", invalid_value_replacement = None, with_data = True, with_statistics = True):
		super(CategoricalDomain, self).__init__(missing_values = missing_values, missing_value_treatment = missing_value_treatment, missing_value_replacement = missing_value_replacement, invalid_value_treatment = invalid_value_treatment, invalid_value_replacement = invalid_value_replacement, with_data = with_data, with_statistics = with_statistics)

	def fit(self, X, y = None):
		X = column_or_1d(X, warn = True)
		if self._empty_fit():
			return self
		mask = self._missing_value_mask(X)
		values, counts = numpy.unique(X[~mask], return_counts = True)
		if self.with_data:
			self.data_ = values
		if self.with_statistics:
			self.counts_ = _count(mask)
			self.discr_stats_ = (values, counts)
		return self

def _interquartile_range(X, axis):
	quartiles = numpy.nanpercentile(X, [25, 75], axis = axis)
	return (quartiles[1] - quartiles[0])

def _abjunction(outlier_mask, missing_value_mask):
	outlier_mask[missing_value_mask] = False
	return outlier_mask

class ContinuousDomain(Domain):

	def __init__(self, missing_values = None, missing_value_treatment = "as_is", missing_value_replacement = None, invalid_value_treatment = "return_invalid", invalid_value_replacement = None, outlier_treatment = "as_is", low_value = None, high_value = None, with_data = True, with_statistics = True):
		super(ContinuousDomain, self).__init__(missing_values = missing_values, missing_value_treatment = missing_value_treatment, missing_value_replacement = missing_value_replacement, invalid_value_treatment = invalid_value_treatment, invalid_value_replacement = invalid_value_replacement, with_data = with_data, with_statistics = with_statistics)
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
		else:
			raise ValueError("Outlier treatment {0} not in {1}".format(outlier_treatment, outlier_treatments))

	def fit(self, X, y = None):
		if self._empty_fit():
			return self
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

	def _outlier_mask(self, X):
		mask = self._missing_value_mask(X)
		result = (numpy.less(X, self.low_value, where = ~mask) | numpy.greater(X, self.high_value, where = ~mask))
		return _abjunction(result, mask)

	def _negative_outlier_mask(self, X):
		mask = self._missing_value_mask(X)
		result = numpy.less(X, self.low_value, where = ~mask)
		return _abjunction(result, mask)

	def _positive_outlier_mask(self, X):
		mask = self._missing_value_mask(X)
		result = numpy.greater(X, self.high_value, where = ~mask)
		return _abjunction(result, mask)

	def transform(self, X):
		if self.outlier_treatment == "as_missing_values":
			mask = self._outlier_mask(X)
			if hasattr(self, "missing_values"):
				if type(self.missing_values) is list:
					raise ValueError()
				X[mask] = self.missing_values
			else:
				X[mask] = None
		elif self.outlier_treatment == "as_extreme_values":
			mask = self._negative_outlier_mask(X)
			X[mask] = self.low_value
			mask = self._positive_outlier_mask(X)
			X[mask] = self.high_value
		return super(ContinuousDomain, self).transform(X)

class TemporalDomain(Domain):

	def __init__(self, missing_values = None, missing_value_treatment = "as_is", missing_value_replacement = None, invalid_value_treatment = "return_invalid", invalid_value_replacement = None):
		super(TemporalDomain, self).__init__(missing_values = missing_values, missing_value_treatment = missing_value_treatment, missing_value_replacement = missing_value_replacement, invalid_value_treatment = invalid_value_treatment, invalid_value_replacement = invalid_value_replacement, with_data = False, with_statistics = False)

	def _to_instant(self, X):
		return X.to_pydatetime()

	def fit(self, X, y = None):
		return self

	def transform(self, X):
		shape = X.shape
		if len(shape) > 1:
			X = X.ravel()
		Xt = pandas.to_datetime(X, yearfirst = True, origin = "unix")
		Xt = self._to_instant(Xt)
		if len(shape) > 1:
			Xt = Xt.reshape(shape)
		return Xt

class DateDomain(TemporalDomain):

	def __init__(self, missing_values = None, missing_value_treatment = "as_is", missing_value_replacement = None, invalid_value_treatment = "return_invalid", invalid_value_replacement = None):
		super(DateDomain, self).__init__(missing_values = missing_values, missing_value_treatment = missing_value_treatment, missing_value_replacement = missing_value_replacement, invalid_value_treatment = invalid_value_treatment, invalid_value_replacement = invalid_value_replacement)

	def _to_instant(self, X):
		return super(DateDomain, self)._to_instant(X.floor("D"))

class DateTimeDomain(TemporalDomain):

	def __init__(self, missing_values = None, missing_value_treatment = "as_is", missing_value_replacement = None, invalid_value_treatment = "return_invalid", invalid_value_replacement = None):
		super(DateTimeDomain, self).__init__(missing_values = missing_values, missing_value_treatment = missing_value_treatment, missing_value_replacement = missing_value_replacement, invalid_value_treatment = invalid_value_treatment, invalid_value_replacement = invalid_value_replacement)

	def _to_instant(self, X):
		return super(DateTimeDomain, self)._to_instant(X.floor("S"))

class MultiDomain(BaseEstimator, TransformerMixin):

	def __init__(self, domains):
		self.domains = domains

	def fit(self, X, y = None):
		rows, columns = X.shape
		if len(self.domains) != columns:
			raise ValueError("The number of columns {0} is not equal to the number of domain objects {1}".format(columns, len(self.domains)))
		if isinstance(X, DataFrame):
			for domain, column in zip(self.domains, X.columns):
				domain.fit(X[column].values)
		else:
			for domain, column in zip(self.domains, range(0, columns)):
				domain.fit(X[:, column])
		return self

	def transform(self, X):
		rows, columns = X.shape
		if len(self.domains) != columns:
			raise ValueError("The number of columns {0} is not equal to the number of domain objects {1}".format(columns, len(self.domains)))
		if isinstance(X, DataFrame):
			for domain, column in zip(self.domains, X.columns):
				X[column] = domain.transform(X[column].values)
		else:
			for domain, column in zip(self.domains, range(0, columns)):
				X[:, column] = domain.transform(X[:, column])
		return X
