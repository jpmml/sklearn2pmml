from pandas import DataFrame
from sklearn2pmml.util import sizeof, deep_sizeof, Evaluatable, Slicer, Reshaper
from unittest import TestCase

import numpy

class MeasurementTest(TestCase):

	def test_sizeof(self):
		self.assertEqual(sizeof(None, with_overhead = False), sizeof(None, with_overhead = True))

		self.assertEqual(sizeof(1, with_overhead = False), sizeof(1, with_overhead = True))
		self.assertEqual(sizeof(True, with_overhead = False), sizeof(True, with_overhead = True))

		self.assertEqual(sizeof(1), sizeof(2))
		self.assertEqual(sizeof(1.0), sizeof(2.0))

		self.assertTrue(sizeof([], with_overhead = False) < sizeof([], with_overhead = True))

		self.assertTrue(sizeof(["a"]) > sizeof([]))
		self.assertTrue(sizeof(["abc"], with_overhead = False) == sizeof(["a"], with_overhead = False))
		self.assertTrue(sizeof(["abc"], with_overhead = True) == sizeof(["a"], with_overhead = True))
		self.assertTrue(sizeof(["aaa", "bbb", "ccc"], with_overhead = False) == sizeof(["a", "b", "c"], with_overhead = False))
		self.assertTrue(sizeof(["aaa", "bbb", "ccc"], with_overhead = True) == sizeof(["a", "b", "c"], with_overhead = True))

		self.assertTrue(sizeof((), with_overhead = False) < sizeof((), with_overhead = True))

		self.assertTrue(sizeof(("a")) > sizeof(()))
		self.assertTrue(sizeof(("abc")) == sizeof(("a")) + 2)
		self.assertTrue(sizeof(("aaa", "bbb", "ccc"), with_overhead = False) == sizeof(("a", "b", "c"), with_overhead = False))
		self.assertTrue(sizeof(("aaa", "bbb", "ccc"), with_overhead = True) == sizeof(("a", "b", "c"), with_overhead = True))

	def test_deep_sizeof(self):
		self.assertTrue(deep_sizeof([], with_overhead = False) < deep_sizeof([], with_overhead = True))

		self.assertEqual(deep_sizeof(["abc"], with_overhead = False), deep_sizeof(["a"], with_overhead = False) + 2)
		self.assertEqual(deep_sizeof(["abc"], with_overhead = True), deep_sizeof(["a"], with_overhead = True) + 2)
		self.assertEqual(deep_sizeof(["aaa", "bbb", "ccc"], with_overhead = False), deep_sizeof(["a", "b", "c"], with_overhead = False) + 6)
		self.assertEqual(deep_sizeof(["aaa", "bbb", "ccc"], with_overhead = True), deep_sizeof(["a", "b", "c"], with_overhead = True) + 6)

		self.assertTrue(deep_sizeof((), with_overhead = False) < deep_sizeof((), with_overhead = True))

		self.assertEqual(deep_sizeof(("aaa", "bbb", "ccc"), with_overhead = False), deep_sizeof(("a", "b", "c"), with_overhead = False) + 6)
		self.assertEqual(deep_sizeof(("aaa", "bbb", "ccc"), with_overhead = True), deep_sizeof(("a", "b", "c"), with_overhead = True) + 6)

class EvaluatableTest(TestCase):

	def test_setup_and_evaluate(self):
		isNegativeDef = "def _is_negative(x):\n	return (x < 0)\n"
		isPositiveDef = "def _is_positive(x):\n	return (x > 0)\n"
		expr = "_is_negative(X[0])"
		evaluatable = Evaluatable(expr, function_defs = [])
		with self.assertRaises(NameError):
			evaluatable.setup_and_evaluate([-1.5], env = dict())
		evaluatable = Evaluatable(expr, function_defs = [isNegativeDef])
		env = dict()
		self.assertTrue(evaluatable.setup_and_evaluate([-1.5], env = env))
		self.assertFalse(evaluatable.setup_and_evaluate([1.5], env = env))
		expr = "_is_negative(X[0]) or _is_positive(X[0])"
		evaluatable = Evaluatable(expr, function_defs = [isNegativeDef])
		with self.assertRaises(NameError):
			evaluatable.setup_and_evaluate([0], env = dict())
		evaluatable = Evaluatable(expr, function_defs = [isNegativeDef, isPositiveDef])
		env = dict()
		self.assertTrue(evaluatable.setup_and_evaluate([-1.5], env = env))
		self.assertFalse(evaluatable.setup_and_evaluate([0], env = env))
		self.assertTrue(evaluatable.setup_and_evaluate([1.5], env = env))

		signumDef = "def _signum(x):\n\timport numpy\n\tif _is_negative(x):\n\t\treturn numpy.ceil(-1.5)\n\telif _is_positive(x):\n\t\treturn numpy.floor(1.5)\n\telse:\n\t\treturn 0\n"
		expr = "_signum(X[0])"
		evaluatable = Evaluatable(expr, function_defs = [signumDef, isNegativeDef, isPositiveDef])
		env = dict()
		self.assertEqual(-1, evaluatable.setup_and_evaluate([-1.5], env = env))
		self.assertEqual(0, evaluatable.setup_and_evaluate([0], env = env))
		self.assertEqual(1, evaluatable.setup_and_evaluate([1.5], env = env))

class ReshaperTest(TestCase):

	def test_transform(self):
		transformer = Reshaper((1, 6))
		X = numpy.asarray([[0, "zero"], [1, "one"], [2, "two"]], dtype = object)
		Xt = transformer.fit_transform(X)
		self.assertEqual([[0, "zero", 1, "one", 2, "two"]], Xt.tolist())
		transformer = Reshaper((6, 1))
		Xt = transformer.fit_transform(X)
		self.assertEqual([[0], ["zero"], [1], ["one"], [2], ["two"]], Xt.tolist())

class SlicerTest(TestCase):

	def test_transform(self):
		transformer = Slicer()
		X = DataFrame([[1.5, False, 0], [1.0, True, 1], [0.5, False, 0]], columns = ["a", "b", "c"])
		Xt = transformer.fit_transform(X)
		self.assertEqual((3, 3), Xt.shape)
		self.assertEqual(["a", "b", "c"], Xt.columns.tolist())
		transformer = Slicer(start = 1)
		Xt = transformer.fit_transform(X)
		self.assertEqual((3, 2), Xt.shape)
		self.assertEqual(["b", "c"], Xt.columns.tolist())
		transformer = Slicer(stop = -1)
		Xt = transformer.fit_transform(X)
		self.assertEqual((3, 2), Xt.shape)
		self.assertEqual(["a", "b"], Xt.columns.tolist())
		transformer = Slicer(start = 1, stop = -1)
		Xt = transformer.fit_transform(X)
		self.assertIsInstance(Xt, DataFrame)
		self.assertEqual((3, 1), Xt.shape)
		self.assertEqual(["b"], Xt.columns.tolist())
		X = X.values
		Xt = transformer.fit_transform(X)
		self.assertIsInstance(Xt, numpy.ndarray)
		self.assertEqual((3, 1), Xt.shape)
		self.assertEqual([[False], [True], [False]], Xt.tolist())
