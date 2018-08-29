from h2o.frame import H2OFrame
from sklearn.base import BaseEstimator, TransformerMixin

class H2OFrameCreator(BaseEstimator, TransformerMixin):

	def __init__(self, column_names = None, column_types = None):
		self.column_names = column_names
		self.column_types = column_types

	def fit(self, X, y = None):
		return self

	def transform(self, X):
		return H2OFrame(python_obj = X, column_names = self.column_names, column_types = self.column_types)
