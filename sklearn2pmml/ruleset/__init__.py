from sklearn.base import BaseEstimator, ClassifierMixin

import numpy

class RuleSetClassifier(BaseEstimator, ClassifierMixin):

	def __init__(self, rules, default_score = None):
		for rule in rules:
			if type(rule) is not tuple:
				raise ValueError("Rule is not a tuple")
			if len(rule) < 2:
				raise ValueError("Rule is not a two-element (predicate, score) tuple")
		self.rules = rules
		self.default_score = default_score

	def fit(self, X, y = None):
		return self

	def score(self, X):
		for rule in self.rules:
			predicate = rule[0]
			score = rule[1]
			if eval(predicate):
				return score
		return self.default_score

	def predict(self, X):
		func = lambda x: self.score(x)
		if hasattr(X, "apply"):
			return X.apply(func, axis = 1)
		nrow = X.shape[0]
		y = numpy.empty(shape = (nrow, ), dtype = object)
		for i in range(0, nrow):
			y[i] = func(X[i])
		return y
