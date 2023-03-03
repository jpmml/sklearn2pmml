from pandas import DataFrame
from sklearn2pmml.expression import ExpressionRegressor
from sklearn2pmml.util import Expression
from unittest import TestCase

import numpy

def _relu(x):
	if x > 0:
		return x
	else:
		return 0

class ExpressionRegressorTest(TestCase):

	def test_predict(self):
		regressor = ExpressionRegressor(Expression("_relu(X[0])", function_defs = [_relu]))
		X = numpy.array([[-2], [-1], [0], [1], [2]], dtype = int)
		y = regressor.predict(X)
		self.assertEqual(float, y.dtype)
		self.assertEqual([0, 0, 0, 1, 2], y.tolist())
		X = DataFrame([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], columns = ["x"])
		y = regressor.predict(X)
		self.assertEqual([0.0, 0.0, 0.0, 0.5, 1.5, 2.5], y.tolist())
