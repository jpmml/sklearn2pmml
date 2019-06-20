from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn2pmml.util import eval_rows

class RuleSetClassifier(BaseEstimator, ClassifierMixin):

	def __init__(self, rules, default_score = None):
		for rule in rules:
			if type(rule) is not tuple:
				raise ValueError("Rule is not a tuple")
			if len(rule) != 2:
				raise ValueError("Rule is not a two-element (predicate, score) tuple")
		self.rules = rules
		self.default_score = default_score

	def _eval_row(self, X):
		for rule in self.rules:
			predicate = rule[0]
			score = rule[1]
			if eval(predicate):
				return score
		return self.default_score

	def fit(self, X, y = None):
		return self

	def predict(self, X):
		func = lambda x: self._eval_row(x)
		return eval_rows(X, func)
