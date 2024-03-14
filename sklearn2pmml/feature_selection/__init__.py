from pandas import DataFrame
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn2pmml.util import to_numpy

import numpy

class SelectUnique(BaseEstimator, SelectorMixin):

	def __init__(self):
		pass

	def _get_support_mask(self):
		check_is_fitted(self, "support_mask_")
		return self.support_mask_

	def fit(self, X, y = None):
		X = to_numpy(X)
		rows, cols = X.shape
		mask = numpy.full((cols), fill_value = True)
		for left in range(cols):
			if mask[left] is False:
				continue
			for right in range(left + 1, cols):
				equiv = numpy.array_equiv(X[:, left], X[:, right])
				if equiv:
					mask[right] = False
		self.support_mask_ = mask
		return self