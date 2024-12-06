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
		pred = classifier.predict(X)
		self.assertEqual(["zero", "positive", "negative"], pred.tolist())
		X = DataFrame(X, columns = ["x"])
		pred = classifier.predict(X)
		self.assertEqual(["zero", "positive", "negative"], pred.tolist())
		classifier = RuleSetClassifier([
			("X['x'] < 0", "negative"),
			("X['x'] > 0", "positive")
		], default_score = "zero")
		pred = classifier.predict(X)
		self.assertEqual(["zero", "positive", "negative"], pred.tolist())
