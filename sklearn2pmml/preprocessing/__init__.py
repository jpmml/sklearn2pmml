from pandas import Series
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import column_or_1d

import numpy
import pandas

class PMMLLabelBinarizer(BaseEstimator, TransformerMixin):

	def fit(self, y):
		y = column_or_1d(y, warn = True)
		self.classes_ = numpy.unique(y[~pandas.isnull(y)])
		return self

	def transform(self, y):
		y = column_or_1d(y, warn = True)
		index = list(self.classes_)
		Y = numpy.zeros((len(y), len(index)), dtype = numpy.int)
		for i, v in enumerate(y):
			if not pandas.isnull(v):
				Y[i, index.index(v)] = 1
		return Y

class PMMLLabelEncoder(BaseEstimator, TransformerMixin):

	def __init__(self, missing_value = None):
		self.missing_value = missing_value

	def fit(self, y):
		y = column_or_1d(y, warn = True)
		self.classes_ = numpy.unique(y[~pandas.isnull(y)])
		return self

	def transform(self, y):
		y = column_or_1d(y, warn = True)
		index = list(self.classes_)
		return numpy.array([self.missing_value if pandas.isnull(v) else index.index(v) for v in y])
