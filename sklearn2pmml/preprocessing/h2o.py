try:
	from h2o.frame import H2OFrame
except ImportError:
	pass
from sklearn.base import BaseEstimator, TransformerMixin

import warnings

class H2OFrameConstructor(BaseEstimator, TransformerMixin):

	def __init__(self, column_names = None, column_types = None):
		self.column_names = column_names
		self.column_types = column_types

	def fit(self, X, y = None):
		return self

	def transform(self, X):
		return H2OFrame(python_obj = X, column_names = self.column_names, column_types = self.column_types)

class H2OFrameCreator(H2OFrameConstructor):

	def __init__(self, column_names = None, column_types = None):
		super(H2OFrameCreator, self).__init__(column_names = column_names, column_types = column_types)
		warnings.warn("{} has been renamed to {}, the alias will be removed in the future".format(H2OFrameCreator.__name__, H2OFrameConstructor.__name__), DeprecationWarning)
