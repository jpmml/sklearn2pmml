try:
	from CHAID import Tree
except ImportError:
	pass
from pandas import Series
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

import numpy
import pandas

class CHAIDEstimator(BaseEstimator):

	def __init__(self, config):
		self.config = config

	def _target_type(self):
		raise NotImplementedError()

	def fit(self, X, y, **fit_params):
		features = dict()
		for col in X.columns.values:
			features[col] = "nominal"
		df = pandas.concat((X, y), axis = 1)
		tree = Tree.from_pandas_df(df, i_variables = features, d_variable = y.name, dep_variable_type = self._target_type(), **self.config)
		treelib_tree = tree.to_tree()
		self.tree_ = tree
		self.treelib_tree_ = treelib_tree
		return self

class CHAIDClassifier(CHAIDEstimator, ClassifierMixin):

	def __init__(self, config = {}):
		super(CHAIDClassifier, self).__init__(config = config)

	def _target_type(self):
		return "categorical"

	def fit(self, X, y, **fit_params):
		classes, y_encoded = numpy.unique(y, return_inverse = True)
		self.classes_ = classes
		y_encoded = Series(y_encoded, name = y.name)
		y = y_encoded
		super(CHAIDClassifier, self).fit(X, y, **fit_params)

	def predict(self, X):
		raise NotImplementedError()

	def predict_proba(self, X):
		raise NotImplementedError()

class CHAIDRegressor(CHAIDEstimator, RegressorMixin):

	def __init__(self, config = {}):
		super(CHAIDRegressor, self).__init__(config = config)

	def _target_type(self):
		return "continuous"

	def predict(self, X):
		raise NotImplementedError()