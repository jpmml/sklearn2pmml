from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn2pmml.util import check_predicate, eval_rows, to_expr_func

class RuleSetClassifier(BaseEstimator, ClassifierMixin):

	def __init__(self, rules, default_score = None):
		for rule in rules:
			if type(rule) is not tuple:
				raise TypeError("Rule is not a tuple")
			if len(rule) != 2:
				raise TypeError("Rule is not a two-element (predicate, score) tuple")
			predicate, score = rule
			check_predicate(predicate)
		self.rules = rules
		self.default_score = default_score

	def fit(self, X, y = None):
		return self

	def predict(self, X):
		rules = [(to_expr_func(predicate), score) for (predicate, score) in self.rules]

		def _eval_row(x):
			for (expr_func, score) in rules:
				if expr_func(x):
					return score
			return self.default_score

		return eval_rows(X, _eval_row)
