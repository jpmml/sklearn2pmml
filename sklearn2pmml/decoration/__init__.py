from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import column_or_1d

import numpy
import pandas

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
		missing_value_treatments = ["as_is", "as_mean", "as_mode", "as_median", "as_value"]
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

	def _get_mask(self, X):
		if hasattr(self, "missing_values"):
			return X == self.missing_values
		else:
			return pandas.isnull(X)

	def transform(self, X):
		if hasattr(self, "missing_value_replacement"):
			mask = self._get_mask(X)
			X[mask] = self.missing_value_replacement
		return X

class CategoricalDomain(Domain):

	def __init__(self, **kwargs):
		super(CategoricalDomain, self).__init__(**kwargs)

	def fit(self, X, y = None):
		X = column_or_1d(X, warn = True)
		if self._empty_fit():
			return self
		mask = self._get_mask(X)
		values, counts = numpy.unique(X[~mask], return_counts = True)
		if self.with_data:
			self.data_ = values
		if self.with_statistics:
			self.counts_ = _count(mask)
			self.discr_stats_ = (values, counts)
		return self

class ContinuousDomain(Domain):

	def __init__(self, **kwargs):
		super(ContinuousDomain, self).__init__(**kwargs)

	def _interquartile_range(self, X, axis):
		quartiles = numpy.nanpercentile(X, [25, 75], axis = axis)
		return (quartiles[1] - quartiles[0])

	def fit(self, X, y = None):
		if self._empty_fit():
			return self
		mask = self._get_mask(X)
		X = numpy.ma.masked_array(X, mask = mask)
		min = numpy.asarray(numpy.nanmin(X, axis = 0))
		max = numpy.asarray(numpy.nanmax(X, axis = 0))
		if self.with_data:
			self.data_min_ = min
			self.data_max_ = max
		if self.with_statistics:
			self.counts_ = _count(mask)
			X = numpy.ma.asarray(X, dtype = numpy.float).filled(float('NaN'))
			self.numeric_info_ = {
				"minimum" : min,
				"maximum" : max,
				"mean" : numpy.asarray(numpy.nanmean(X, axis = 0)),
				"standardDeviation" : numpy.asarray(numpy.nanstd(X, axis = 0)),
				"median" : numpy.asarray(numpy.nanmedian(X, axis = 0)),
				"interQuartileRange" : numpy.asarray(self._interquartile_range(X, axis = 0))
			}
		return self
