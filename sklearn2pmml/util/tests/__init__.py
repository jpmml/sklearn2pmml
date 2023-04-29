from pandas import DataFrame
from sklearn2pmml.util import sizeof, deep_sizeof, to_expr, to_expr_func, fqn, Evaluatable, Slicer, Reshaper
from unittest import TestCase

import inspect
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

def _is_missing(X):
	return pandas.isnull(X[0])

def _signum(X):
	if X[0] < 0:
		return -1
	elif X[0] > 0:
		return 1
	else:
		return 0

def _is_negative(x):
	return (_trunc(x) < 0)

def _is_positive(x):
	return (_trunc(x) > 0)

def _trunc(x):
	return math.trunc(x)

class Dummy:
	pass

class FunctionTest(TestCase):

	def test_inline_expr(self):
		expr = "pandas.isnull(X[0])"
		expr = to_expr(expr)
		expr_func = to_expr_func(expr, modules = [])
		with self.assertRaises(NameError):
			expr_func([1.5])
		expr_func = to_expr_func(expr, modules = ["pandas"])
		self.assertFalse(expr_func([1.5]))
		self.assertTrue(expr_func([numpy.NaN]))

		expr = "-1 if (X[0] < 0) else (1 if (X[0] > 0) else 0)"
		expr = to_expr(expr)
		expr_func = to_expr_func(expr, modules = [])
		self.assertEqual(-1, expr_func([-1.5]))
		self.assertEqual(0, expr_func([0]))
		self.assertEqual(1, expr_func([1.5]))

	def test_inline_def_expr(self):
		expr = _is_missing
		expr = to_expr(expr)
		self.assertIsInstance(expr, str)
		expr_func = to_expr_func(expr, modules = [])
		with self.assertRaises(NameError):
			expr_func([1.5])
		expr_func = to_expr_func(expr, modules = ["pandas"])
		self.assertFalse(expr_func([1.5]))
		self.assertTrue(expr_func([numpy.NaN]))

		expr = inspect.getsource(_signum)
		expr = to_expr(expr)
		self.assertIsInstance(expr, str)
		expr_func = to_expr_func(expr, modules = [])
		self.assertEqual(-1, expr_func([-1.5]))
		self.assertEqual(0, expr_func([0]))
		self.assertEqual(1, expr_func([1.5]))

	def test_evaluatable_expr(self):
		expr = Evaluatable("_is_negative(X[0])")
		expr = to_expr(expr)
		self.assertIsInstance(expr, Evaluatable)
		expr_func = to_expr_func(expr)
		with self.assertRaises(NameError):
			expr_func([-1.5])
		expr = Evaluatable("_is_negative(X[0])", function_defs = [_is_negative])
		expr = to_expr(expr)
		expr_func = to_expr_func(expr)
		with self.assertRaises(NameError):
			expr_func([-1.5])
		expr = Evaluatable("_is_negative(X[0])", function_defs = [_is_negative, _trunc])
		expr = to_expr(expr)
		expr_func = to_expr_func(expr)
		self.assertTrue(expr_func([-1.5]))
		self.assertFalse(expr_func([0]))
		self.assertFalse(expr_func([1.5]))

		expr = Evaluatable("-1 if _is_negative(X[0]) else (1 if _is_positive(X[0]) else 0)", function_defs = [_is_negative, _is_positive, _trunc])
		expr = to_expr(expr)
		self.assertIsInstance(expr, Evaluatable)
		expr_func = to_expr_func(expr)
		self.assertEqual(-1, expr_func([-1.5]))
		self.assertEqual(0, expr_func([0]))
		self.assertEqual(1, expr_func([1.5]))

	def test_fqn(self):
		self.assertEqual("builtins.type", fqn(str))

		self.assertEqual("builtins.str", fqn(""))

		self.assertEqual("builtins.type", fqn(Dummy))
		self.assertEqual("builtins.type", fqn(Dummy.__class__))

		dummy = Dummy()
		self.assertEqual("sklearn2pmml.util.tests.Dummy", fqn(dummy))

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
