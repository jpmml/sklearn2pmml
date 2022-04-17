from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn2pmml.preprocessing import ExpressionTransformer

class BusinessDecisionTransformer(BaseEstimator, TransformerMixin):

	def __init__(self, transformer, business_problem, decisions, prefit = False):
		self.transformer = transformer
		self.business_problem = business_problem
		for decision in decisions:
			if type(decision) is not tuple:
				raise ValueError("Decision is not a tuple")
			if len(decision) != 2:
				raise ValueError("Decision is not a two-element (value, description) tuple")
		self.decisions = decisions
		self.prefit = prefit
		if prefit:
			self.transformer_ = clone(transformer)

	def fit(self, X, y = None):
		self.transformer_ = clone(self.transformer)
		if y is None:
			self.transformer_.fit(X)
		else:
			self.transformer_.fit(X, y)
		return self

	def transform(self, X):
		return self.transformer_.transform(X)
