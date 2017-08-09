#!/usr/bin/env python

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import column_or_1d

import numpy
import pandas

__copyright__ = "Copyright (c) 2016 Villu Ruusmann"
__license__ = "GNU Affero General Public License (AGPL) version 3.0"

class Domain(BaseEstimator, TransformerMixin):

	def __init__(self, missing_value_treatment = "as_is", missing_value_replacement = None, invalid_value_treatment = "return_invalid"):
		missing_value_treatments = ["as_is", "as_mean", "as_mode", "as_median", "as_value"]
		if missing_value_treatment not in missing_value_treatments:
			raise ValueError("Missing value treatment {0} not in {1}".format(missing_value_treatment, missing_value_treatments))
		self.missing_value_treatment = missing_value_treatment
		if(missing_value_replacement is not None):
			self.missing_value_replacement = missing_value_replacement
		invalid_value_treatments = ["return_invalid", "as_is", "as_missing"]
		if invalid_value_treatment not in invalid_value_treatments:
			raise ValueError("Invalid value treatment {0} not in {1}".format(invalid_value_treatment, invalid_value_treatments))
		self.invalid_value_treatment = invalid_value_treatment

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
		self.data_ = numpy.unique(X[~pandas.isnull(X)])
		self.data_total_freq_ = X.size
		self.data_missing_freq_ = numpy.sum(pandas.isnull(X).astype(int), axis=0)
		return self

class ContinuousDomain(Domain):

	def __init__(self, **kwargs):
		Domain.__init__(self, **kwargs)

	def fit(self, X, y = None):
		self.data_min_ = numpy.nanmin(X, axis = 0)
		self.data_max_ = numpy.nanmax(X, axis = 0)
		self.data_mean_ = numpy.nanmean(X, axis = 0)
		self.data_std_ = numpy.nanstd(X, axis = 0)
		self.data_median_ = numpy.nanmedian(X, axis = 0)
		self.data_inter_quartile_range_ = numpy.nanpercentile(X, 75, axis = 0) - numpy.nanpercentile(X, 25, axis = 0)
		self.data_1st_quantile_ = numpy.nanpercentile(X, 25, axis = 0)
		self.data_3rd_quantile_ = numpy.nanpercentile(X, 75, axis = 0)
		self.data_total_freq_ = numpy.empty(X.shape[1])
		self.data_total_freq_.fill(X.shape[0])
		self.data_missing_freq_ = numpy.array(numpy.sum(numpy.isnan(X).astype(int), axis=0))
		return self
