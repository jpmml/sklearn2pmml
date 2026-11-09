from pandas import Categorical, CategoricalDtype, DataFrame, Series
from sklearn.utils.validation import check_is_fitted
from sklearn2pmml.util import _is_categorical, _is_ordinal, _is_pandas_proto_categorical, _eval_array_rows, _eval_scalar_rows, _get_column_count, _get_column_names, _to_1d, check_expression, check_predicate, fqn, to_evaluatable_func, to_expr, Evaluatable, Expression, Predicate, Slicer, Reshaper
from unittest import TestCase

import inspect
import numpy
import polars

class ContainerTest(TestCase):

	def test_column_count(self):
		self.assertEqual(1, _get_column_count([0, 2, 1]))
		X = numpy.asarray([0, 2, 1])
		self.assertEqual(1, _get_column_count(X))
		X = numpy.asarray([[0, -1], [2, 0], [1, 1]])
		self.assertEqual(2, _get_column_count(X))
		X = numpy.asarray([[0], [2], [1]])
		self.assertEqual(1, _get_column_count(X))
		X = Categorical(["a", "b", "c"])
		self.assertEqual(1, _get_column_count(X))
		X = Series([0, 2, 1])
		self.assertEqual(1, _get_column_count(X))
		X = DataFrame([[0], [2], [1]], columns = ["a"])
		self.assertEqual(1, _get_column_count(X))
		X = DataFrame([[0, False], [2, True], [1, True]], columns = ["a", "b"])
		self.assertEqual(2, _get_column_count(X))

	def test_column_names(self):
		X = DataFrame([[1, 0], [2, 0], [3, 0]], columns = [1, 2])
		self.assertEqual(["1", "2"], _get_column_names(X).tolist())
		X.columns = numpy.asarray([1.0, 2.0])
		self.assertEqual(["1.0", "2.0"], _get_column_names(X).tolist())
		X = Series([1, 2, 3], name = 1)
		self.assertEqual("1", _get_column_names(X).tolist())
		X.name = 1.0
		self.assertEqual("1.0", _get_column_names(X).tolist())

	def test_polars_column_names(self):
		X = polars.DataFrame([[1, 0], [2, 0], [3, 0]], schema = ["1", "2"], orient = "row")
		self.assertEqual(["1", "2"], _get_column_names(X).tolist())
		X = polars.Series(name = "1", values = [1, 2, 3])
		self.assertEqual("1", _get_column_names(X).tolist())

	def test_to_1d(self):
		X = Series([1, 2, 3])
		Xt = _to_1d(X)
		self.assertIs(X, Xt)
		X = DataFrame([[1], [2], [3]], columns = ["a"])
		self.assertEqual((3, 1), X.shape)
		Xt = _to_1d(X)
		self.assertIsInstance(Xt, Series)
		self.assertEqual([1, 2, 3], Xt.tolist())

	def test_polars_to_1d(self):
		X = polars.Series(name = "a", values = [1, 2, 3])
		Xt = _to_1d(X)
		self.assertIs(X, Xt)
		X = polars.DataFrame([[1], [2], [3]], schema = ["a"], orient = "row")
		self.assertEqual((3, 1), X.shape)
		Xt = _to_1d(X)
		self.assertIsInstance(Xt, polars.Series)
		self.assertEqual([1, 2, 3], Xt.to_list())

class DTypeTest(TestCase):

	def test_is_categorical(self):
		x = Series(["True", "False", "True"], name = "x", dtype = str)
		self.assertEqual(["True", "False", "True"], x.values.tolist())
		self.assertTrue(_is_categorical(x.dtype))
		x = Series([True, False, True], name = "x", dtype = bool)
		self.assertEqual([True, False, True], x.values.tolist())
		self.assertTrue(_is_categorical(x.dtype))
		x = x.astype(float)
		self.assertEqual([1.0, 0.0, 1.0], x.values.tolist())
		self.assertFalse(_is_categorical(x.dtype))
		x = x.astype("category")
		self.assertEqual([1.0, 0.0, 1.0], x.values.tolist())
		self.assertTrue(_is_categorical(x.dtype))
		x = x.astype(int)
		self.assertEqual([1, 0, 1], x.values.tolist())
		self.assertFalse(_is_categorical(x.dtype))
		x = x.astype(CategoricalDtype())
		self.assertEqual([1, 0, 1], x.values.tolist())
		self.assertTrue(_is_categorical(x.dtype))

	def test_is_pandas_proto_categorical(self):
		dtype = "category"
		self.assertTrue(_is_pandas_proto_categorical(dtype))
		dtype = CategoricalDtype()
		self.assertTrue(_is_pandas_proto_categorical(dtype))
		dtype = CategoricalDtype(categories = [])
		self.assertFalse(_is_pandas_proto_categorical(dtype))
		dtype = CategoricalDtype(categories = ["a", "b", "c"])
		self.assertFalse(_is_pandas_proto_categorical(dtype))

	def test_is_ordinal(self):
		x = Categorical(["True", "False", "True"], categories = ["True", "False"], ordered = True)
		self.assertTrue(_is_ordinal(x.dtype))
		x = Categorical([True, False, True], categories = [True, False], ordered = True)
		self.assertTrue(_is_ordinal(x.dtype))
		x = Series(["True", "False", "True"], dtype = "category")
		self.assertFalse(_is_ordinal(x.dtype))
		self.assertEqual([1, 0, 1], x.cat.codes.tolist())
		x = x.astype(CategoricalDtype(categories = ["True", "False"], ordered = True))
		self.assertTrue(_is_ordinal(x.dtype))
		self.assertEqual([0, 1, 0], x.cat.codes.tolist())

def _is_missing(X):
	return pandas.isnull(X[0])

def _signum(X):
	if X[0] < 0:
		return -1
	elif X[0] > 0:
		return 1
	else:
		return 0

class Dummy:
	pass

class FunctionTest(TestCase):

	def test_check_expression(self):
		self.assertIsNotNone(check_expression("1.0"))
		self.assertIsNotNone(check_expression(Expression("1.0")))
		with self.assertRaises(TypeError):
			check_expression(Predicate(str(True)))

	def test_check_predicate(self):
		self.assertIsNotNone(check_predicate(str(True)))
		self.assertIsNotNone(check_predicate(str(True)))
		with self.assertRaises(TypeError):
			check_predicate(Expression("1.0"))

	def test_eval_array_rows(self):
		X = Series([-1, 0, 1], index = [10, 20, 30])
		Xt = _eval_array_rows(X, to_evaluatable_func("X[0] > 0"), dtype = bool)
		self.assertEqual([10, 20, 30], Xt.index.tolist())
		self.assertEqual([False, False, True], Xt.tolist())
		X = DataFrame([[-1], [0], [1]], columns = ["a"], index = [10, 20, 30])
		Xt = _eval_array_rows(X, to_evaluatable_func("X['a'] > 0"), dtype = bool)
		self.assertEqual([10, 20, 30], Xt.index.tolist())
		self.assertEqual([False, False, True], Xt.tolist())

	def test_eval_scalar_rows(self):
		X = Series([-1, 0, 1], index = [10, 20, 30])
		Xt = _eval_scalar_rows(X, lambda x: x > 0, dtype = bool)
		self.assertEqual([10, 20, 30], Xt.index.tolist())
		self.assertEqual([False, False, True], Xt.tolist())
		X = numpy.asarray([-1, 0, 1])
		Xt = _eval_scalar_rows(X, lambda x: x > 0, dtype = bool)
		self.assertEqual([False, False, True], Xt.tolist())

	def test_fqn(self):
		obj = ""
		self.assertEqual("builtins.str", fqn(str))
		self.assertEqual("builtins.type", fqn(str.__class__))
		self.assertEqual("builtins.str", fqn(obj))

		obj = Dummy()
		self.assertEqual("test_util.Dummy", fqn(Dummy))
		self.assertEqual("builtins.type", fqn(Dummy.__class__))
		self.assertEqual("test_util.Dummy", fqn(obj))

	def test_inline_expr(self):
		expr = "-1 if (X[0] < 0) else (1 if (X[0] > 0) else 0)"
		expr = to_expr(expr)
		expr_func = to_evaluatable_func(expr, modules = [])
		self.assertEqual(-1, expr_func([-1.5]))
		self.assertEqual(0, expr_func([0]))
		self.assertEqual(1, expr_func([1.5]))

		expr = "re.search(X[0], X[1])"
		expr = to_expr(expr)
		expr_func = to_evaluatable_func(expr)
		self.assertTrue(expr_func(["ar?y", "February"]))
		self.assertFalse(expr_func(["ar?y", "March"]))
		self.assertTrue(expr_func(["ar?y", "May"]))

		expr = "pandas.isnull(X[0])"
		expr = to_expr(expr)
		expr_func = to_evaluatable_func(expr, modules = [])
		with self.assertRaises(NameError):
			expr_func([1.5])
		expr_func = to_evaluatable_func(expr, modules = ["pandas"])
		self.assertFalse(expr_func([1.5]))
		self.assertTrue(expr_func([numpy.nan]))

		expr = "mymodule.myfunc(X[0])"
		expr = to_expr(expr)
		expr_func = to_evaluatable_func(expr)
		with self.assertRaises(NameError):
			expr_func([0])

	def test_inline_def_expr(self):
		expr = _is_missing
		expr = to_expr(expr)
		self.assertIsInstance(expr, str)
		expr_func = to_evaluatable_func(expr, modules = [])
		with self.assertRaises(NameError):
			expr_func([1.5])
		expr_func = to_evaluatable_func(expr, modules = ["pandas"])
		self.assertFalse(expr_func([1.5]))
		self.assertTrue(expr_func([numpy.nan]))

		expr = inspect.getsource(_signum)
		expr = to_expr(expr)
		self.assertIsInstance(expr, str)
		expr_func = to_evaluatable_func(expr, modules = [])
		self.assertEqual(-1, expr_func([-1.5]))
		self.assertEqual(0, expr_func([0]))
		self.assertEqual(1, expr_func([1.5]))

def _is_negative(x):
	return (_trunc(x) < 0)

def _is_positive(x):
	return (_trunc(x) > 0)

def _trunc(x):
	return math.trunc(x)

class EvaluatableTest(TestCase):

	def test_evaluatable_expr(self):
		expr = Evaluatable("_is_negative(X[0])")
		expr = to_expr(expr)
		self.assertIsInstance(expr, Evaluatable)
		expr_func = to_evaluatable_func(expr)
		with self.assertRaises(NameError):
			expr_func([-1.5])
		expr = Evaluatable("_is_negative(X[0])", function_defs = [_is_negative])
		expr = to_expr(expr)
		self.assertFalse(expr.uses("math"))
		self.assertFalse(expr.uses("numpy"))
		expr_func = to_evaluatable_func(expr)
		with self.assertRaises(NameError):
			expr_func([-1.5])
		expr = Evaluatable("_is_negative(X[0])", function_defs = [_is_negative, _trunc])
		expr = to_expr(expr)
		self.assertTrue(expr.uses("math"))
		self.assertFalse(expr.uses("numpy"))
		expr_func = to_evaluatable_func(expr)
		self.assertTrue(expr_func([-1.5]))
		self.assertFalse(expr_func([0]))
		self.assertFalse(expr_func([1.5]))

		expr = Evaluatable("-1 if _is_negative(X[0]) else (1 if _is_positive(X[0]) else 0)", function_defs = [_is_negative, _is_positive, _trunc])
		expr = to_expr(expr)
		self.assertIsInstance(expr, Evaluatable)
		expr_func = to_evaluatable_func(expr)
		self.assertEqual(-1, expr_func([-1.5]))
		self.assertEqual(0, expr_func([0]))
		self.assertEqual(1, expr_func([1.5]))

class ReshaperTest(TestCase):

	def test_transform(self):
		transformer = Reshaper((1, 6))
		check_is_fitted(transformer)
		X = numpy.asarray([[0, "zero"], [1, "one"], [2, "two"]], dtype = object)
		Xt = transformer.fit_transform(X)
		self.assertEqual(2, transformer.n_features_in_)
		self.assertEqual([[0, "zero", 1, "one", 2, "two"]], Xt.tolist())
		transformer = Reshaper((6, 1))
		Xt = transformer.fit_transform(X)
		self.assertEqual([[0], ["zero"], [1], ["one"], [2], ["two"]], Xt.tolist())

class SlicerTest(TestCase):

	def test_transform(self):
		transformer = Slicer()
		check_is_fitted(transformer)
		X = DataFrame([[1.5, False, 0], [1.0, True, 1], [0.5, False, 0]], columns = ["a", "b", "c"])
		Xt = transformer.fit_transform(X)
		self.assertEqual(3, transformer.n_features_in_)
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
