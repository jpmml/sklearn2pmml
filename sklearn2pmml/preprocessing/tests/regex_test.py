from pandas import DataFrame, Series
from sklearn2pmml.preprocessing.regex import MatchesTransformer, ReplaceTransformer
from unittest import TestCase

import numpy

class MatchesTransformerTest(TestCase):

	def test_transform(self):
		transformer = MatchesTransformer("ar?y")
		y = numpy.asarray(["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])
		yt = transformer.transform(y)
		self.assertEqual([True, True, False, False, True, False, False, False, False, False, False, False], yt.tolist())
		y = DataFrame(y.reshape(-1, 1), columns = ["month"])
		yt = transformer.transform(y)
		self.assertTrue((12, 1), yt.shape)
		self.assertEqual([True, True, False, False, True, False, False, False, False, False, False, False], yt.tolist())

class ReplaceTransformerTest(TestCase):

	def test_transform(self):
		transformer = ReplaceTransformer("B+", "c")
		y = numpy.asarray(["A", "B", "BA", "BB", "BAB", "ABBA", "BBBB"])
		yt = transformer.transform(y)
		self.assertEqual(["A", "c", "cA", "c", "cAc", "AcA", "c"], yt.tolist())
		y = DataFrame(y.reshape(-1, 1), columns = ["input"])
		self.assertTrue((7, 1), yt.shape)
		self.assertEqual(["A", "c", "cA", "c", "cAc", "AcA", "c"], yt.tolist())