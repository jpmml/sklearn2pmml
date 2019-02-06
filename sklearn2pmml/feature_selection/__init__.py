from pandas import DataFrame
from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin
from sklearn.utils.validation import check_is_fitted

import numpy

class SelectUnique(BaseEstimator, SelectorMixin):

	def __init__(self):
		pass

	def _get_support_mask(self):
		check_is_fitted(self, "support_mask_")
		return self.support_mask_

	def fit(self, X, y = None):
		rows, cols = X.shape
		mask = numpy.full((cols), True, dtype = bool)
		if isinstance(X, DataFrame):
			X = X.values
		for left in range(cols):
			if mask[left] is False:
				continue
			for right in range(left + 1, cols):
				equiv = numpy.array_equiv(X[:, left], X[:, right])
				if(equiv):
					mask[right] = False
		self.support_mask_ = mask
		return self