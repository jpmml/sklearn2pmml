from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameConstructor(BaseEstimator, TransformerMixin):

	def __init__(self, columns, dtype):
		self.columns = columns
		self.dtype = dtype

	def fit(self, X, y = None):
		return self

	def transform(self, X):
		return DataFrame(X, columns = self.columns, dtype = self.dtype)
