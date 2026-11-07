from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn2pmml.base import StatelessEstimatorMixin
from sklearn2pmml.util import _eval_array_rows, check_predicate, to_evaluatable_func

class RuleSetClassifier(BaseEstimator, StatelessEstimatorMixin, ClassifierMixin):

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
		rules = [(to_evaluatable_func(predicate), score) for (predicate, score) in self.rules]

		def eval_func(x):
			for (predicate_func, score) in rules:
				if predicate_func(x):
					return score
			return self.default_score

		return _eval_array_rows(X, eval_func)
