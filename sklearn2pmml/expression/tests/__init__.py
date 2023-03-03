from pandas import DataFrame
from scipy.special import expit
from sklearn2pmml.expression import ExpressionClassifier, ExpressionRegressor
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
		X = numpy.asarray([[-2], [-1], [0], [1], [2]], dtype = int)
		y = regressor.predict(X)
		self.assertEqual(float, y.dtype)
		self.assertEqual([0, 0, 0, 1, 2], y.tolist())
		X = DataFrame([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], columns = ["x"])
		y = regressor.predict(X)
		self.assertEqual([0.0, 0.0, 0.0, 0.5, 1.5, 2.5], y.tolist())

def _is_negative(x):
	if x < 0:
		return 1
	else:
		return 0

def _is_neutral(x):
	if (x >= -0.5) and (x <= 0.5):
		return 2
	else:
		return 0

def _is_positive(x):
	if x > 0:
		return 1
	else:
		return 0

class ExpressionClassifierTest(TestCase):

	def test_binary_fit_predict(self):
		class_exprs = {
			"yes": Expression("X[0] ** X[1] + X[2]")
		}
		classifier = ExpressionClassifier(class_exprs, normalization_method = "logit")
		X = numpy.asarray([[2, 1/2, 0], [2, 1, 0], [2, 2, -6]])
		y = numpy.asarray(["yes", "yes", "no"])
		classifier.fit(X, y)
		self.assertEqual(["no", "yes"], classifier.classes_.tolist())
		pred_dfs = classifier.decision_function(X)
		self.assertEqual((3, ), pred_dfs.shape)
		self.assertEqual([numpy.sqrt(2), 2, -2], pred_dfs.tolist())
		preds = classifier.predict(X)
		self.assertEqual(["yes", "yes", "no"], preds.tolist())
		pred_probs = classifier.predict_proba(X)
		self.assertEqual((3, 2), pred_probs.shape)
		self.assertAlmostEqual(3, numpy.sum(pred_probs))
		self.assertEqual([expit(numpy.sqrt(2)), expit(2), expit(-2)], pred_probs[:, 1].tolist())

		class_exprs["no"] = Expression("1.0")
		classifier = ExpressionClassifier(class_exprs, normalization_method = "softmax")
		classifier.fit(X, y)
		self.assertEqual(["no", "yes"], classifier.classes_.tolist())
		pred_dfs = classifier.decision_function(X)
		self.assertEqual((3, 2), pred_dfs.shape)
		self.assertEqual([1, 1, 1], pred_dfs[:, 0].tolist())
		self.assertEqual([numpy.sqrt(2), 2, -2], pred_dfs[:, 1].tolist())
		preds = classifier.predict(X)
		self.assertEqual(["yes", "yes", "no"], preds.tolist())
		pred_probs = classifier.predict_proba(X)
		self.assertEqual((3, 2), pred_probs.shape)
		self.assertAlmostEqual(3, numpy.sum(pred_probs))

	def test_multiclass_predict(self):
		class_exprs = {
			"negative" : Expression("_is_negative(X[0])", function_defs = [_is_negative]),
			"neutral" : Expression("_is_neutral(X[0])", function_defs = [_is_neutral]),
			"positive" : Expression("_is_positive(X[0])", function_defs = [_is_positive])
		}
		classifier = ExpressionClassifier(class_exprs, normalization_method = "simplemax")
		X = numpy.asarray([[-0.25], [-1], [1]])
		y = numpy.asarray(["neutral", "negative", "positive"])
		classifier.fit(X, y)
		self.assertEqual(["negative", "neutral", "positive"], classifier.classes_.tolist())
		pred_dfs = classifier.decision_function(X)
		self.assertEqual((3, 3), pred_dfs.shape)
		self.assertEqual([[1, 2, 0], [1, 0, 0], [0, 0, 1]], pred_dfs.tolist())
		preds = classifier.predict(X)
		self.assertEqual(["neutral", "negative", "positive"], preds.tolist())
		pred_probs = classifier.predict_proba(X)
		self.assertEqual((3, 3), pred_probs.shape)
		self.assertAlmostEqual(3, numpy.sum(pred_probs))
		self.assertEqual([[1.0 / 3.0, 2.0 / 3.0, 0], [1, 0, 0], [0, 0, 1]], pred_probs.tolist())

		classifier.normalization_method = "softmax"
		pred_probs = classifier.predict_proba(X)
		self.assertAlmostEqual(3, numpy.sum(pred_probs))
		exp_sum = (numpy.exp(0) + numpy.exp(1) + numpy.exp(2))
		self.assertTrue(numpy.allclose([numpy.exp(1) / exp_sum, numpy.exp(2) / exp_sum, numpy.exp(0) / exp_sum], pred_probs[0, :].tolist()))
		exp_off = (numpy.exp(0) / (2 * numpy.exp(0) + numpy.exp(1)))
		exp_on = (numpy.exp(1) / (2 * numpy.exp(0) + numpy.exp(1)))
		self.assertTrue(numpy.allclose([exp_on, exp_off, exp_off], pred_probs[1, :].tolist()))
		self.assertTrue(numpy.allclose([exp_off, exp_off, exp_on], pred_probs[2, :].tolist()))
