from collections import defaultdict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import column_or_1d

import numpy
import pandas

class ExpressionTransformer(TransformerMixin):

	def __init__(self, expr):
		self.expr_ = expr

	def fit(self, X, y = None):
		return self

	def transform(self, X, y = None):
		return eval(self.expr_)

class LookupTransformer(TransformerMixin):

	def __init__(self, mapping, default_value):
		if type(mapping) is not dict:
			raise ValueError("Input value to output value mapping is not a dict")
		for k, v in mapping.items():
			if k is None:
				raise ValueError("Key is None")
		self.mapping = mapping
		self.default_value = default_value

	def _transform_dict(self):
		transform_dict = defaultdict(lambda: self.default_value)
		transform_dict.update(self.mapping)
		return transform_dict

	def fit(self, y):
		y = column_or_1d(y, warn = True)
		return self

	def transform(self, y):
		y = column_or_1d(y, warn = True)
		transform_dict = self._transform_dict()
		func = lambda k: transform_dict[k]
		if hasattr(y, "apply"):
			return y.apply(func)
		return numpy.vectorize(func)(y)

class MultiLookupTransformer(LookupTransformer):

	def __init__(self, mapping, default_value):
		super(MultiLookupTransformer, self).__init__(mapping, default_value)
		length = -1
		for k, v in mapping.items():
			if type(k) is not tuple:
				raise ValueError("Key is not a tuple")
			if length == -1:
				length = len(k)
				continue
			if length != len(k):
				raise ValueError("Keys contain variable number of elements")

	def fit(self, Y):
		return self

	def transform(self, Y):
		transform_dict = self._transform_dict()
		func = lambda k: transform_dict[tuple(k)]
		if hasattr(Y, "apply"):
			return Y.apply(func, axis = 1)
		# See https://stackoverflow.com/a/3338368
		return numpy.array([func((numpy.squeeze(numpy.asarray(row))).tolist()) for row in Y])

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
