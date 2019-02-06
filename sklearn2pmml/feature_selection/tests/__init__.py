from pandas import DataFrame
from sklearn2pmml.feature_selection import SelectUnique
from unittest import TestCase

import numpy

class SelectUniqueTest(TestCase):

	def test_fit_transform(self):
		selector = SelectUnique()
		X = numpy.asarray([[0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0], [1, -1, 1, 1, -1, 1, -1]])
		Xt = selector.fit_transform(X)
		self.assertEqual([True, True, False, True, False, False, True], selector.support_mask_.tolist())
		self.assertEqual([[0.0, 0.0, 1.0, 1.0], [1, -1, 1, -1]], Xt.tolist())
		X = DataFrame([[1, 1, 0, 1, 0], [True, True, True, True, True]], columns = ["A", "B", "C", "D", "E"])
		Xt = selector.fit_transform(X)
		self.assertEqual([True, False, True, False, False], selector.support_mask_.tolist())
		self.assertEqual([[1, 0], [True, True]], Xt.tolist())