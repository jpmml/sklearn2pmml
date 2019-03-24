from pandas import DataFrame, Series
from sklearn2pmml.preprocessing.regex import MatchesTransformer, ReplaceTransformer
from unittest import TestCase

import numpy

class MatchesTransformerTest(TestCase):

	def test_transform(self):
		transformer = MatchesTransformer("ar?y")
		X = numpy.asarray(["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])
		Xt = transformer.transform(X)
		self.assertEqual([True, True, False, False, True, False, False, False, False, False, False, False], Xt.tolist())
		X = DataFrame(X.reshape(-1, 1), columns = ["month"])
		Xt = transformer.transform(X)
		self.assertTrue((12, 1), Xt.shape)
		self.assertEqual([True, True, False, False, True, False, False, False, False, False, False, False], Xt.tolist())

class ReplaceTransformerTest(TestCase):

	def test_transform(self):
		transformer = ReplaceTransformer("B+", "c")
		X = numpy.asarray(["A", "B", "BA", "BB", "BAB", "ABBA", "BBBB"])
		Xt = transformer.transform(X)
		self.assertEqual(["A", "c", "cA", "c", "cAc", "AcA", "c"], Xt.tolist())
		X = DataFrame(X.reshape(-1, 1), columns = ["input"])
		self.assertTrue((7, 1), Xt.shape)
		self.assertEqual(["A", "c", "cA", "c", "cAc", "AcA", "c"], Xt.tolist())