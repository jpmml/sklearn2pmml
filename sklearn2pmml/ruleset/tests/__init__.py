from pandas import DataFrame
from sklearn2pmml.ruleset import RuleSetClassifier
from unittest import TestCase

import numpy

class RuleSetClassifierTest(TestCase):

	def test_predict(self):
		classifier = RuleSetClassifier([
			("X[0] < 0", "negative"),
			("X[0] > 0", "positive")
		], default_score = "zero")
		X = numpy.array([[0.0], [1.0], [-1.0]])
		y = classifier.predict(X)
		self.assertEqual(["zero", "positive", "negative"], y.tolist())
		X = DataFrame(X, columns = ["x"])
		y = classifier.predict(X)
		self.assertEqual(["zero", "positive", "negative"], y.tolist())
		classifier = RuleSetClassifier([
			("X['x'] < 0", "negative"),
			("X['x'] > 0", "positive")
		], default_score = "zero")
		y = classifier.predict(X)
		self.assertEqual(["zero", "positive", "negative"], y.tolist())
