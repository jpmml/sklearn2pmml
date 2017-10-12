from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import column_or_1d

import numpy
import pandas

def _count(mask):
	if hasattr(mask, "values"):
		mask = mask.values
	non_missing_freq = sum(mask)
	missing_freq = sum(~mask)
	return {
		"totalFreq" : (non_missing_freq + missing_freq),
		"missingFreq" : missing_freq,
		"invalidFreq" : (non_missing_freq - non_missing_freq) # A scalar zero, or an array of zeroes
	}

class Domain(BaseEstimator, TransformerMixin):

	def __init__(self, missing_value_treatment = "as_is", missing_value_replacement = None, invalid_value_treatment = "return_invalid", with_data = True, with_statistics = True):
		missing_value_treatments = ["as_is", "as_mean", "as_mode", "as_median", "as_value"]
		if missing_value_treatment not in missing_value_treatments:
			raise ValueError("Missing value treatment {0} not in {1}".format(missing_value_treatment, missing_value_treatments))
		self.missing_value_treatment = missing_value_treatment
		if missing_value_replacement is not None:
			self.missing_value_replacement = missing_value_replacement
		invalid_value_treatments = ["return_invalid", "as_is", "as_missing"]
		if invalid_value_treatment not in invalid_value_treatments:
			raise ValueError("Invalid value treatment {0} not in {1}".format(invalid_value_treatment, invalid_value_treatments))
		self.invalid_value_treatment = invalid_value_treatment
		self.with_data = with_data
		self.with_statistics = with_statistics

	def transform(self, X):
		if hasattr(self, "missing_value_replacement"):
			if hasattr(X, "fillna"):
				X.fillna(value = self.missing_value_replacement, inplace = True)
			else:
				X[pandas.isnull(X)] = self.missing_value_replacement
		return X

class CategoricalDomain(Domain):

	def __init__(self, **kwargs):
		Domain.__init__(self, **kwargs)

	def fit(self, X, y = None):
		X = column_or_1d(X, warn = True)
		mask = pandas.notnull(X)
		values, counts = numpy.unique(X[mask], return_counts = True)
		if self.with_data:
			self.data_ = values
		if self.with_statistics:
			self.counts_ = _count(mask)
			self.discr_stats_ = (values, counts)
		return self

class ContinuousDomain(Domain):

	def __init__(self, **kwargs):
		Domain.__init__(self, **kwargs)

	def fit(self, X, y = None):
		mask = pandas.notnull(X)
		min = numpy.nanmin(X, axis = 0)
		max = numpy.nanmax(X, axis = 0)
		if self.with_data:
			self.data_min_ = min
			self.data_max_ = max
		if self.with_statistics:
			self.counts_ = _count(mask)
			self.numeric_info_ = {
				"minimum" : min,
				"maximum" : max,
				"mean" : numpy.nanmean(X, axis = 0),
				"standardDeviation" : numpy.nanstd(X, axis = 0),
				"median" : numpy.nanmedian(X, axis = 0),
				"interQuartileRange" : (numpy.nanpercentile(X, 75, axis = 0) - numpy.nanpercentile(X, 25, axis = 0))
			}
		return self
