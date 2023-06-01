from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn2pmml.preprocessing import ExpressionTransformer

import copy

class BusinessDecisionTransformer(BaseEstimator, TransformerMixin):

	def __init__(self, transformer, business_problem, decisions, prefit = True):
		if transformer is None:
			pass
		elif isinstance(transformer, str):
			pass
		elif isinstance(transformer, TransformerMixin):
			pass
		else:
			raise TypeError("The transformer object is not an instance of {0}".format(TransformerMixin.__name__))
		self.transformer = transformer
		self.business_problem = business_problem
		for decision in decisions:
			if type(decision) is not tuple:
				raise TypeError("Decision is not a tuple")
			if len(decision) != 2:
				raise TypeError("Decision is not a two-element (value, description) tuple")
		self.decisions = decisions
		self.prefit = prefit
		if prefit:
			self.transformer_ = self._make_transformer(prefit = True)

	def _make_transformer(self, prefit = True):
		if self.transformer is None:
			return None
		elif isinstance(self.transformer, str):
			return ExpressionTransformer(self.transformer)
		elif isinstance(self.transformer, TransformerMixin):
			if prefit:
				return copy.deepcopy(self.transformer)
			return clone(self.transformer)
		else:
			raise TypeError("The transformer object is not an instance of {0}".format(TransformerMixin.__name__))

	def fit(self, X, y = None):
		self.transformer_ = self._make_transformer(prefit = False)
		if self.transformer_ is not None:
			if y is None:
				self.transformer_.fit(X)
			else:
				self.transformer_.fit(X, y)
		return self

	def transform(self, X):
		if self.transformer_ is not None:
			return self.transformer_.transform(X)
		return X
