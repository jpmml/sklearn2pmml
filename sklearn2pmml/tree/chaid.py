try:
	from CHAID import Tree
except ImportError:
	pass
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

import numpy
import pandas
import warnings

class CHAIDEstimator(BaseEstimator):

	def __init__(self, config):
		self.config = config

	def _target_type(self):
		raise NotImplementedError()

	def fit(self, X, y, **fit_params):
		if isinstance(X, DataFrame):
			features = dict()
			for col in X.columns.values:
				features[col] = "nominal"
			df = pandas.concat((X, y), axis = 1)
			tree = Tree.from_pandas_df(df, i_variables = features, d_variable = y.name, dep_variable_type = self._target_type(), **self.config)
		else:
			variable_types = ["nominal"] * X.shape[1]
			tree = Tree.from_numpy(X, y, variable_types = variable_types, dep_variable_type = self._target_type(), **self.config)
		treelib_tree = tree.to_tree()
		self.tree_ = tree
		self.treelib_tree_ = treelib_tree
		return self

	def apply(self, X):
		if X is not None:
			warnings.warn("Ignoring the 'X' argument. Returning leaf indices for the training dataset")
		return self.tree_.node_predictions()

class CHAIDClassifier(CHAIDEstimator, ClassifierMixin):

	def __init__(self, config = {}):
		super(CHAIDClassifier, self).__init__(config = config)

	def _target_type(self):
		return "categorical"

	def fit(self, X, y, **fit_params):
		classes, y_encoded = numpy.unique(y, return_inverse = True)
		self.classes_ = classes
		y_encoded = Series(y_encoded, name = y.name)
		super(CHAIDClassifier, self).fit(X, y_encoded, **fit_params)
		return self

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