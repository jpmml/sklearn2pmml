from sklearn2pmml.sklearn_pandas import patch_sklearn

patch_sklearn()

from datetime import datetime
from pandas import CategoricalDtype, DataFrame, Series, Timestamp
from sklearn_pandas import DataFrameMapper
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn2pmml.decoration import Alias, DateDomain, DateTimeDomain
from sklearn2pmml.preprocessing import AggregateTransformer, CastTransformer, ConcatTransformer, CutTransformer, DataFrameConstructor, DateTimeFormatter, DaysSinceYearTransformer, ExpressionTransformer, FilterLookupTransformer, IdentityTransformer, LagTransformer, LookupTransformer, MatchesTransformer, MultiCastTransformer, MultiLookupTransformer, NumberFormatter, PMMLLabelBinarizer, PMMLLabelEncoder, PowerFunctionTransformer, ReplaceTransformer, RollingAggregateTransformer, SecondsSinceMidnightTransformer, SecondsSinceYearTransformer, SelectFirstTransformer, SeriesConstructor, StringLengthTransformer, StringNormalizer, SubstringTransformer, WordCountTransformer
from sklearn2pmml.preprocessing.h2o import H2OFrameConstructor, H2OFrameCreator
from sklearn2pmml.preprocessing.lightgbm import make_lightgbm_column_transformer, make_lightgbm_dataframe_mapper
from sklearn2pmml.preprocessing.xgboost import make_xgboost_column_transformer, make_xgboost_dataframe_mapper
from sklearn2pmml.util import to_expr, Expression
from unittest import TestCase

import inspect
import math
import numpy
import pandas
import pcre2
import scipy

def _list_equal(left, right):
	left = DataFrame(left, dtype = object)
	right = DataFrame(right, dtype = object)
	return left.equals(right)

class TransformerTest(TestCase):

	def _fit_transform1d(self, transformer, X):
		transformer = transformer.fit(X)
		return self._transform1d(transformer, X)

	def _transform1d(self, transformer, X):
		self.assertIsInstance(X, Series)
		X_ndarray = X.values
		self.assertIsInstance(X_ndarray, numpy.ndarray)
		Xt = transformer.transform(X)
		Xt_ndarray = transformer.transform(X_ndarray)
		self.assertIsInstance(Xt, Series)
		self.assertIsInstance(Xt_ndarray, numpy.ndarray)
		self.assertEqual(Xt.tolist(), Xt_ndarray.tolist())
		return Xt

class AggregateTransformerTest(TestCase): 

	def test_init(self):
		with self.assertRaises(ValueError):
			AggregateTransformer(None)

	def test_avg_float(self):
		X = numpy.asarray([1.0, float("NaN"), 2.0])
		transformer = AggregateTransformer(function = "avg")
		X = X.reshape((-1, 3))
		self.assertEqual(1.5, transformer.transform(X))

	def test_min_int(self):
		X = numpy.asarray([1, 0, 2, 3])
		transformer = AggregateTransformer(function = "min")
		X = X.reshape((-1, 4))
		self.assertEqual(0, transformer.transform(X))
		X = X.reshape((2, 2))
		self.assertEqual([[0], [2]], transformer.transform(X).tolist())
		X = X.reshape((4, -1))
		self.assertEqual([[1], [0], [2], [3]], transformer.transform(X).tolist())

	def test_min_float(self):
		X = numpy.asarray([1.0, 0.5, 2.0, 3.0, float("NaN"), 1.5])
		transformer = AggregateTransformer(function = "min")
		X = X.reshape((-1, 6))
		self.assertEqual(0.5, transformer.transform(X))
		X = X.reshape((3, 2))
		self.assertEqual([[0.5], [2.0], [1.5]], transformer.transform(X).tolist())
		X = X.reshape((2, 3))
		self.assertEqual([[0.5], [1.5]], transformer.transform(X).tolist())
		X = X.reshape((6, -1))
		self.assertTrue(_list_equal([[1.0], [0.5], [2.0], [3.0], [float("NaN")], [1.5]], transformer.transform(X).tolist()))

	def test_product_float(self):
		X = numpy.asarray([1.0, float("NaN"), 2.0, 4.0])
		transformer = AggregateTransformer(function = "product")
		X = X.reshape((-1, 4))
		self.assertEqual(8.0, transformer.transform(X))
		X = X.reshape((2, 2))
		self.assertEqual([[1.0], [8.0]], transformer.transform(X).tolist())

	def test_sum_float(self):
		X = numpy.asarray([1.0, float("NaN"), 2.0, 1.0])
		transformer = AggregateTransformer(function = "sum")
		X = X.reshape((-1, 4))
		self.assertEqual(4.0, transformer.transform(X))
		X = X.reshape((2, 2))
		self.assertEqual([[1.0], [3.0]], transformer.transform(X).tolist())

class CastTransformerTest(TransformerTest):

	def test_transform(self):
		X = Series([False, "1", float(1.0), 0], dtype = object)
		transformer = CastTransformer(dtype = str)
		self.assertEqual(["False", "1", "1.0", "0"], self._fit_transform1d(transformer, X).tolist())
		transformer = CastTransformer(dtype = int)
		self.assertEqual([0, 1, 1, 0], self._fit_transform1d(transformer, X).tolist())
		transformer = CastTransformer(dtype = float)
		self.assertEqual([0.0, 1.0, 1.0, 0.0], self._fit_transform1d(transformer, X).tolist())
		transformer = CastTransformer(dtype = numpy.float64)
		self.assertEqual([0.0, 1.0, 1.0, 0.0], self._fit_transform1d(transformer, X).tolist())
		transformer = CastTransformer(dtype = bool)
		self.assertEqual([False, True, True, False], self._fit_transform1d(transformer, X).tolist())

	def test_transform_datetime(self):
		X = Series(["1969-07-16T13:32:00Z".replace("Z", ""), datetime.fromisoformat("1969-07-20T20:17:40Z".replace("Z", "")), Timestamp("1969-07-21T17:54:01Z").tz_localize(None), "1969-07-24T16:50:35Z".replace("Z", "")], dtype = str)
		transformer = CastTransformer(dtype = "datetime64[D]")
		self.assertEqual([datetime(1969, 7, 16), datetime(1969, 7, 20), datetime(1969, 7, 21), datetime(1969, 7, 24)], self._fit_transform1d(transformer, X).tolist())
		transformer = CastTransformer(dtype = "datetime64[s]")
		self.assertEqual([datetime(1969, 7, 16, 13, 32, 0), datetime(1969, 7, 20, 20, 17, 40), datetime(1969, 7, 21, 17, 54, 1), datetime(1969, 7, 24, 16, 50, 35)], self._fit_transform1d(transformer, X).tolist())

	def test_transform_string(self):
		X = Series([float("NaN"), "red", None, "green", pandas.NA], dtype = object)
		transformer = CastTransformer(dtype = str)
		self.assertTrue(_list_equal([float("NaN"), "red", None, "green", pandas.NA], self._fit_transform1d(transformer, X).tolist()))

	def test_transform_categorical(self):
		X = Series(["a", "c", "b", "a", "a", "b"], name = "x")
		transformer = CastTransformer(dtype = "category")
		self.assertTrue(hasattr(transformer, "dtype"))
		self.assertFalse(hasattr(transformer, "dtype_"))
		Xt = transformer.fit_transform(X)
		self.assertTrue(hasattr(transformer, "dtype"))
		self.assertTrue(hasattr(transformer, "dtype_"))
		self.assertEqual(["a", "b", "c"], transformer.dtype_.categories.tolist())
		X = X.values
		transformer = CastTransformer(dtype = "category")
		# Can fit, but cannot transform Numpy arrays
		transformer.fit(X)
		self.assertEqual(["a", "b", "c"], transformer.dtype_.categories.tolist())

class MultiCastTransformerTest(TestCase):

	def test_transform(self):
		X = DataFrame([["-1", "-1.0"], ["0", "0.0"], ["1", "1.0"]], columns = ["int", "float"])
		transformer = MultiCastTransformer(dtypes = [int, float])
		self.assertEqual([int, float], transformer.dtypes)
		self.assertFalse(hasattr(transformer, "dtypes_"))
		Xt = transformer.fit_transform(X)
		self.assertEqual([int, float], transformer.dtypes_)
		self.assertIsInstance(Xt, DataFrame)
		self.assertEqual([[-1, -1.0], [0, 0.0], [1, 1.0]], Xt.values.tolist())
		X = X.to_numpy()
		Xt = transformer.transform(X)
		self.assertIsInstance(Xt, numpy.ndarray)
		self.assertEqual([[-1, -1.0], [0, 0.0], [1, 1.0]], Xt.tolist())

	def test_transform_categorical(self):
		X = DataFrame([["apple", "green"], ["cherry", "red"], ["apple", "red"], ["banana", "yellow"], ["avocado", "green"]], columns = ["fruit", "color"])
		transformer = MultiCastTransformer(dtypes = ["category", pandas.CategoricalDtype()])
		self.assertFalse(hasattr(transformer, "dtypes_"))
		Xt = transformer.fit_transform(X)
		fruit_dtype = transformer.dtypes_[0]
		color_dtype = transformer.dtypes_[1]
		self.assertIsInstance(fruit_dtype, CategoricalDtype)
		self.assertEqual(["apple", "avocado", "banana", "cherry"], fruit_dtype.categories.tolist())
		self.assertIsInstance(fruit_dtype, CategoricalDtype)
		self.assertEqual(["green", "red", "yellow"], color_dtype.categories.tolist())

class CutTransformerTest(TestCase):

	def test_transform_int(self):
		X = DataFrame([[-25], [-1], [5], [25]], columns = ["a"])
		bins = [-100, -10, 0, 10, 100]
		transformer = CutTransformer(bins, labels = ["x-neg", "neg", "pos", "x-pos"], dtype = "category")
		Xt = transformer.fit_transform(X)
		self.assertIsInstance(Xt, Series)
		self.assertEqual(["x-neg", "neg", "pos", "x-pos"], Xt.tolist())
		transformer = CutTransformer(bins, labels = ["x-neg", "neg", "pos", "x-pos"])
		Xt = transformer.fit_transform(X)
		self.assertIsInstance(Xt, numpy.ndarray)
		self.assertEqual([["x-neg"], ["neg"], ["pos"], ["x-pos"]], Xt.tolist())
		X = numpy.array([-25, -1, 5, 25])
		Xt = transformer.fit_transform(X)
		self.assertIsInstance(Xt, numpy.ndarray)
		self.assertEqual(["x-neg", "neg", "pos", "x-pos"], Xt.tolist())

	def test_transform_float(self):
		bins = [float("-inf"), -1.0, 0.0, 1.0, float("+inf")]
		transformer = CutTransformer(bins, labels = False, right = True)
		X = numpy.array([-2.0, -1.0, -0.5, 0.0, float("NaN"), 0.5, 1.0, 2.0])
		self.assertTrue(_list_equal([[0], [0], [1], [1], [float("NaN")], [2], [2], [3]], transformer.transform(X).tolist()))
		transformer = CutTransformer(bins, labels = False, right = False)
		self.assertTrue(_list_equal([[0], [1], [1], [2], [float("NaN")], [2], [3], [3]], transformer.transform(X).tolist()))
		bins = [-3.0, -1.0, 1.0, 3.0]
		transformer = CutTransformer(bins, labels = False, right = True, include_lowest = True)
		X = numpy.array([-3.0, -2.0, float("NaN"), 2.0, 3.0])
		self.assertTrue(_list_equal([[0], [0], [float("NaN")], [2], [2]], transformer.transform(X).tolist()))
		X = numpy.array([-5.0, 5.0])
		self.assertTrue(_list_equal([[float("NaN")], [float("NaN")]], transformer.transform(X).tolist()))
		bins = [float("-inf"), float("+inf")]
		transformer = CutTransformer(bins, labels = ["any"])
		self.assertEqual(["any", "any"], transformer.transform(X).tolist())

class DataFrameConstructorTest(TestCase):

	def test_fit_transform(self):
		transformer = DataFrameConstructor(columns = ["int", "str"], dtype = object)
		X = numpy.asarray([[1, "one"], [2, "two"], [3, "three"]])
		Xt = transformer.fit_transform(X)
		self.assertIsInstance(Xt, DataFrame)
		self.assertEqual(["int", "str"], Xt.columns.values.tolist())
		self.assertEqual(object, Xt["int"].dtype)
		self.assertEqual(object, Xt["str"].dtype)

	def test_get_feature_names_out(self):
		transformer = DataFrameConstructor(columns = ["x1", "x2"], dtype = int)
		self.assertEqual(["x1", "x2"], transformer.get_feature_names_out().tolist())
		X = numpy.asarray([[0, 0], [0, 1], [1, 1]])
		pipeline = Pipeline([
			("transformer", transformer)
		])
		if hasattr(pipeline, "set_output"):
			pipeline.set_output(transform = None)
			Xt = pipeline.fit_transform(X)
			self.assertIsInstance(Xt, DataFrame)
			self.assertEqual(["x1", "x2"], Xt.columns.tolist())

class SeriesConstructorTest(TestCase):

	def test_fit_transform(self):
		transformer = SeriesConstructor(name = "x", dtype = object)
		X = numpy.asarray(["one", "two", "three", "one", "three"])
		Xt = transformer.fit_transform(X)
		self.assertIsInstance(Xt, Series)
		self.assertEqual("x", Xt.name)
		self.assertEqual(object, Xt.dtype)
		transformer = SeriesConstructor(name = "x", dtype = "category")
		Xt = transformer.fit_transform(X)
		self.assertIsInstance(Xt.dtype, CategoricalDtype)
		self.assertEqual(["one", "three", "two"], Xt.dtype.categories.tolist())
		transformer = SeriesConstructor(name = "x", dtype = CategoricalDtype(["one", "two"]))
		Xt = transformer.fit_transform(X)
		self.assertIsInstance(Xt.dtype, CategoricalDtype)
		self.assertEqual(["one", "two"], Xt.dtype.categories.tolist())
		self.assertTrue(_list_equal(["one", "two", float("NaN"), "one", float("NaN")], Xt.tolist()))

	def test_get_feature_names_out(self):
		transformer = SeriesConstructor(name = None, dtype = int)
		X = numpy.asarray([0, 2, 1])
		Xt = transformer.fit_transform(X)
		self.assertIsInstance(Xt, Series)
		self.assertEqual(None, Xt.name)
		pipeline = Pipeline([
			("transformer", transformer)
		])
		if hasattr(pipeline, "set_output"):
			pipeline.set_output(transform = None)
			Xt = pipeline.fit_transform(X)
			self.assertIsInstance(Xt, Series)
			self.assertEqual(None, Xt.name)
			pipeline.set_output(transform = "pandas")
			Xt = pipeline.fit_transform(X)
			self.assertIsInstance(Xt, DataFrame)
			self.assertEqual([0], Xt.columns.tolist())
		transformer = SeriesConstructor(name = "flag", dtype = int)
		Xt = transformer.fit_transform(X)
		self.assertIsInstance(Xt, Series)
		self.assertEqual("flag", Xt.name)
		self.assertEqual(["flag"], transformer.get_feature_names_out().tolist())
		pipeline = Pipeline([
			("transformer", transformer)
		])
		if hasattr(pipeline, "set_output"):
			pipeline.set_output(transform = None)
			Xt = pipeline.fit_transform(X)
			self.assertIsInstance(Xt, Series)
			self.assertEqual("flag", Xt.name)
			pipeline.set_output(transform = "pandas")
			Xt = pipeline.fit_transform(X)
			self.assertIsInstance(Xt, DataFrame)
			self.assertEqual(["flag"], Xt.columns.tolist())

class DurationTransformerTest(TestCase):

	def test_days_transform(self):
		X = numpy.array([datetime(1960, 1, 1), datetime(1960, 1, 2), datetime(1960, 2, 1), datetime(1959, 12, 31), datetime(2003, 4, 1)])
		transformer = DaysSinceYearTransformer(year = 1960)
		self.assertEqual([0, 1, 31, -1, 15796], transformer.transform(X).tolist())

	def test_seconds_transform(self):
		X = numpy.array([datetime(1960, 1, 1), datetime(1960, 1, 1, 0, 0, 1), datetime(1960, 1, 1, 0, 1, 0), datetime(1959, 12, 31, 23, 59, 59), datetime(1960, 1, 3, 3, 30, 3)])
		transformer = SecondsSinceYearTransformer(year = 1960)
		self.assertEqual([0, 1, 60, -1, 185403], transformer.transform(X).tolist())

	def test_timedelta_days(self):
		X = DataFrame([["2018-12-31", "2019-01-01"], ["2019-01-31", "2019-01-01"]], columns = ["left", "right"])
		pipeline = clone(Pipeline([
			("union", FeatureUnion([
				("left_mapper", DataFrameMapper([
					("left", [DateDomain(), DaysSinceYearTransformer(year = 2010)])
				])),
				("right_mapper", DataFrameMapper([
					("right", [DateDomain(), DaysSinceYearTransformer(year = 2010)])
				]))
			])),
			("expression", Alias(ExpressionTransformer("X[0] - X[1]"), "delta(left, right)", prefit = True))
		]))
		Xt = pipeline.fit_transform(X)
		self.assertEqual([[-1], [30]], Xt.tolist())

	def test_timedelta_seconds(self):
		X = DataFrame([["2018-12-31T23:59:59", "2019-01-01T00:00:00"], ["2019-01-01T03:30:03", "2019-01-01T00:00:00"]], columns = ["left", "right"])
		mapper = DataFrameMapper([
			(["left", "right"], [DateTimeDomain(), SecondsSinceYearTransformer(year = 2010), ExpressionTransformer("X['left'] - X['right']")])
		], input_df = True)
		Xt = mapper.fit_transform(X)
		self.assertEqual([[-1], [12603]], Xt.tolist())

class SecondsSinceMidnightTransformerTest(TestCase):

	def test_timedelta(self):
		X = DataFrame([["2020-01-01"], ["2020-01-01T00:01:00"], ["2020-01-01T05:23:30"]], columns = ["timestamp"])
		mapper = DataFrameMapper([
			(["timestamp"], [DateTimeDomain(), SecondsSinceMidnightTransformer()])
		])
		Xt = mapper.fit_transform(X)
		self.assertEqual([[0], [60], [19410]], Xt.tolist())

def _signum(X):
	if X[0] < 0: return -1
	elif X[0] > 0: return 1
	else: return 0

def _is_negative(x):
	return (x < 0)

def _is_positive(x):
	return (x > 0)

class ExpressionTransformerTest(TestCase):

	def test_transform(self):
		begin_err_state = numpy.geterr()
		transformer = clone(ExpressionTransformer("X[0]"))
		self.assertTrue(hasattr(transformer, "expr"))
		self.assertTrue(hasattr(transformer, "map_missing_to"))
		self.assertTrue(hasattr(transformer, "default_value"))
		self.assertTrue(hasattr(transformer, "invalid_value_treatment"))
		self.assertTrue(hasattr(transformer, "dtype"))
		self.assertFalse(hasattr(transformer, "dtype_"))
		transformer = ExpressionTransformer("X['a'] + X['b']", map_missing_to = -1, dtype = int)
		X = DataFrame([[0, 1], [1, 2]], columns = ["a", "b"])
		Xt = transformer.fit_transform(X)
		self.assertEqual(int, transformer.dtype)
		self.assertEqual(int, transformer.dtype_)
		self.assertIsInstance(Xt, numpy.ndarray)
		self.assertEqual(int, Xt.dtype)
		self.assertEqual([[1], [3]], Xt.tolist())
		X.iloc[1, 1] = None
		self.assertEqual([[1], [-1]], transformer.fit_transform(X).tolist())
		X = DataFrame([[0.5, 0.5], [1.0, 2.0]], columns = ["a", "b"])
		Xt = transformer.fit_transform(X)
		self.assertIsInstance(Xt, numpy.ndarray)
		self.assertEqual(int, Xt.dtype)
		self.assertEqual([[1], [3]], Xt.tolist())
		X.iloc[1, 1] = float("NaN")
		self.assertEqual([[1], [-1]], transformer.fit_transform(X).tolist())
		transformer = ExpressionTransformer("X['a'] + X['b']", default_value = -1, dtype = float)
		X = DataFrame([[0.5, 0.5], [1.0, 2.0]], columns = ["a", "b"])
		Xt = transformer.fit_transform(X)
		self.assertEqual(float, transformer.dtype)
		self.assertEqual(float, transformer.dtype_)
		self.assertIsInstance(Xt, numpy.ndarray)
		self.assertEqual(float, Xt.dtype)
		self.assertEqual([[1.0], [3.0]], Xt.tolist())
		X.iloc[1, 1] = float("NaN")
		self.assertEqual([[1.0], [-1.0]], transformer.fit_transform(X).tolist())
		transformer = ExpressionTransformer("X[0] + X[1]")
		self.assertTrue(hasattr(transformer, "expr"))
		self.assertTrue(hasattr(transformer, "dtype"))
		self.assertIsNone(transformer.dtype)
		X = numpy.array([[0.5, 0.5], [1.0, 2.0]])
		Xt = transformer.fit_transform(X)
		self.assertEqual(None, transformer.dtype)
		self.assertEqual(None, transformer.dtype_)
		self.assertIsInstance(Xt, numpy.ndarray)
		self.assertEqual([[1], [3]], Xt.tolist())
		transformer = ExpressionTransformer("X[0] - X[1]")
		self.assertEqual([[0.0], [-1.0]], transformer.fit_transform(X).tolist())
		transformer = ExpressionTransformer("X[0] * X[1]")
		self.assertEqual([[0.25], [2.0]], transformer.fit_transform(X).tolist())
		transformer = ExpressionTransformer("X[0] / X[1]")
		self.assertEqual([[1.0], [0.5]], transformer.fit_transform(X).tolist())
		transformer = ExpressionTransformer("X[0] / X[1]", invalid_value_treatment = "return_invalid")
		X = numpy.array([[13, 0]], dtype = int)
		with self.assertRaises(ArithmeticError):
			transformer.transform(X)
		X = X.astype(float)
		with self.assertRaises(ArithmeticError):
			transformer.transform(X)
		transformer = ExpressionTransformer("X[0] / X[1]", invalid_value_treatment = "as_missing")
		X = numpy.array([[13, 0], [13, 1]], dtype = int)
		self.assertEqual([[None], [13]], transformer.fit_transform(X).tolist())
		transformer = ExpressionTransformer("X[0] / X[1]", default_value = -1, invalid_value_treatment = "as_missing")
		self.assertEqual([[-1], [13]], transformer.transform(X).tolist())
		end_err_state = numpy.geterr()
		self.assertEqual(begin_err_state, end_err_state)

	def test_category_transform(self):
		begin_err_state = numpy.geterr()
		transformer = ExpressionTransformer("numpy.rint(X[0])", dtype = "category")
		self.assertTrue(hasattr(transformer, "dtype"))
		self.assertFalse(hasattr(transformer, "dtype_"))
		X = DataFrame([[1.2], [0.1], [1.8], [2.3], [0.7], [0.0]], columns = ["a"])
		with self.assertRaises(NotFittedError):
			transformer.transform(X)
		Xt = transformer.fit_transform(X)
		self.assertTrue(hasattr(transformer, "dtype"))
		self.assertTrue(hasattr(transformer, "dtype_"))
		self.assertIsInstance(transformer.dtype, str)
		self.assertIsInstance(transformer.dtype_, CategoricalDtype)
		self.assertEqual([0, 1, 2], transformer.dtype_.categories.tolist())
		self.assertIsInstance(Xt, Series)
		self.assertIsInstance(Xt.dtype, CategoricalDtype)
		self.assertEqual([1.0, 0.0, 2.0, 2.0, 1.0, 0.0], Xt.tolist())
		end_err_state = numpy.geterr()
		self.assertEqual(begin_err_state, end_err_state)
		transformer = ExpressionTransformer("numpy.rint(X[0])", dtype = CategoricalDtype(categories = [0, 1]))
		self.assertTrue(hasattr(transformer, "dtype"))
		self.assertFalse(hasattr(transformer, "dtype_"))
		Xt = transformer.fit_transform(X)
		self.assertTrue(hasattr(transformer, "dtype"))
		self.assertTrue(hasattr(transformer, "dtype_"))
		self.assertIs(transformer.dtype, transformer.dtype_)
		self.assertIsInstance(Xt, Series)
		self.assertIsInstance(Xt.dtype, CategoricalDtype)
		self.assertTrue(_list_equal([1.0, 0.0, float("NaN"), float("NaN"), 1.0, 0.0], Xt.tolist()))

	def test_func_transform(self):
		expr = to_expr(_signum)
		transformer = clone(ExpressionTransformer(expr))
		X = numpy.array([[1.5], [0.0], [-3.0]])
		self.assertEqual([[1], [0], [-1]], transformer.fit_transform(X).tolist())
		expr = inspect.getsource(_signum).replace("_signum", "_my_signum")
		transformer = clone(ExpressionTransformer(expr))
		self.assertEqual([[1], [0], [-1]], transformer.fit_transform(X).tolist())

		expr = Expression("-1 if _is_negative(X[0]) else (1 if _is_positive(X[0]) else 0)", function_defs = [_is_negative, _is_positive])
		transformer = clone(ExpressionTransformer(expr))
		X = numpy.array([[1.5], [0.0], [-3.0]])
		self.assertEqual([[1], [0], [-1]], transformer.fit_transform(X).tolist())

	def test_regex_transform(self):
		from packaging.version import parse
		# PCRE2 0.4.0
		if parse(pcre2.__version__) == parse("0.4.0"):
			transformer = ExpressionTransformer("pcre2.substitute(r'B+', r'c', X[0])")
		# PCRE2 0.5.0+
		else:
			transformer = ExpressionTransformer("pcre2.sub(r'B+', r'c', X[0])")
		X = numpy.array([["ABBA"]])
		self.assertEqual([["AcA"]], transformer.fit_transform(X).tolist())
		transformer = ExpressionTransformer("re.sub(r'B+', r'c', X[0])")
		self.assertEqual([["AcA"]], transformer.fit_transform(X).tolist())

	def test_sequence_transform(self):
		X = DataFrame([[None], [1], [None]], columns = ["a"])
		mapper = DataFrameMapper([
			(["a"], [ExpressionTransformer("0 if pandas.isnull(X[0]) else X[0]"), SimpleImputer(missing_values = 0)])
		])
		Xt = mapper.fit_transform(X)
		self.assertEqual([[1], [1], [1]], Xt.tolist())

class DateTimeFormatterTest(TransformerTest):

	def test_transform(self):
		transformer = DateTimeFormatter(pattern = "%m/%d/%y")
		self.assertTrue(hasattr(transformer, "pattern"))
		X = DataFrame(["2004-08-20", "2004-08-21T03:15:00"], columns = ["timestamp"])
		mapper = DataFrameMapper([
			(["timestamp"], [DateTimeDomain(), transformer])
		], input_df = True)
		Xt = mapper.fit_transform(X)
		self.assertEqual([["08/20/04"], ["08/21/04"]], Xt.tolist())

class NumberFormatterTest(TransformerTest):

	def test_transform(self):
		transformer = NumberFormatter(pattern = "%3d")
		self.assertTrue(hasattr(transformer, "pattern"))
		X = Series([-1, 0, 1.5])
		self.assertEqual([" -1", "  0", "  1"], self._transform1d(transformer, X).tolist())
		X = numpy.asarray([[-1], [0], [1.5]])
		Xt = transformer.fit_transform(X)
		self.assertIsInstance(Xt, numpy.ndarray)
		self.assertEqual([[" -1"], ["  0"], ["  1"]], Xt.tolist())

class IdentityTransformerTest(TestCase):

	def test_transform(self):
		transformer = IdentityTransformer()
		X = DataFrame(["A", "B", "C"])
		Xt = transformer.fit_transform(X)
		self.assertIsInstance(Xt, DataFrame)
		self.assertEqual([["A"], ["B"], ["C"]], Xt.values.tolist())

class LookupTransformerTest(TransformerTest):

	def test_transform_float(self):
		mapping = {
			int(0.0) : math.cos(0.0),
			45.0 : math.cos(45.0),
			90.0 : math.cos(90.0)
		}
		with self.assertRaises(TypeError):
			LookupTransformer(mapping, None)
		mapping[0.0] = mapping.pop(int(0.0))
		try:
			LookupTransformer(mapping, None)
		except TypeError:
			assert False
		with self.assertRaises(TypeError):
			LookupTransformer(mapping, int(0))
		transformer = LookupTransformer(mapping, float("NaN"))
		X = Series([0.0, 45.0, 90.0])
		self.assertEqual([math.cos(0.0), math.cos(45.0), math.cos(90.0)], self._transform1d(transformer, X).tolist())
		X = numpy.array([float("NaN"), 180.0])
		self.assertTrue(_list_equal([[float("NaN")], [float("NaN")]], transformer.transform(X).tolist()))
		transformer = LookupTransformer(mapping, -999.0)
		self.assertTrue(_list_equal([[float("NaN")], [-999.0]], transformer.transform(X).tolist()))

	def test_transform_string(self):
		mapping = {
			None : "null",
			"one" : "ein",
			"two" : "zwei",
			"three" : "drei"
		}
		with self.assertRaises(ValueError):
			LookupTransformer(mapping, None)
		mapping.pop(None)
		transformer = LookupTransformer(mapping, None)
		X = Series(["one", "two", "three"])
		self.assertEqual(["ein", "zwei", "drei"], self._transform1d(transformer, X).tolist())
		X = numpy.array([[None], ["zero"]])
		self.assertEqual([[None], [None]], transformer.transform(X).tolist())
		transformer = LookupTransformer(mapping, "(other)")
		self.assertEqual([[None], ["(other)"]], transformer.transform(X).tolist())

	def test_transform_categorical(self):
		X = Series(["one", "two", "three"])
		mapping = {
			"one" : "ein",
			"two" : "zwei",
			"three" : "drei"
		}
		transformer = LookupTransformer(mapping, None)
		Xt = transformer.fit_transform(X)
		self.assertIsInstance(Xt, Series)
		self.assertEqual(object, Xt.dtype)
		self.assertEqual(["ein", "zwei", "drei"], Xt.tolist())
		transformer = LookupTransformer(mapping, None, dtype = "category")
		Xt = transformer.fit_transform(X)
		self.assertIsInstance(Xt, Series)
		self.assertIsInstance(Xt.dtype, CategoricalDtype)
		self.assertEqual(["drei", "ein", "zwei"], Xt.dtype.categories.tolist())
		self.assertEqual(["ein", "zwei", "drei"], Xt.tolist())

class FilterLookupTransformerTest(TransformerTest):

	def test_transform_int(self):
		mapping = {
			0 : 1,
			1 : "1",
			2 : 1,
		}
		with self.assertRaises(TypeError):
			FilterLookupTransformer(mapping)
		mapping.pop(1)
		transformer = FilterLookupTransformer(mapping)
		X = Series([0, -1, 3, 2])
		self.assertEqual([1, -1, 3, 1], self._transform1d(transformer, X).tolist())

	def test_transform_string(self):
		mapping = {
			"orange" : "yellow",
			"blue" : None
		}
		with self.assertRaises(ValueError):
			FilterLookupTransformer(mapping)
		mapping["blue"] = "green"
		transformer = FilterLookupTransformer(mapping)
		X = Series(["red", "orange", None, "green", "blue"])
		self.assertEqual(["red", "yellow", None, "green", "green"], self._transform1d(transformer, X).tolist())

class MultiLookupTransformerTest(TestCase):

	def test_transform_int(self):
		mapping = {
			(1, 1) : "one",
			(2, 2) : "two",
			(3, 3) : "three"
		}
		transformer = MultiLookupTransformer(mapping, None)
		X = DataFrame([[1, 0], [1, 1], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1], [3, 2], [3, 3]])
		self.assertEqual([[None], ["one"], [None], [None], ["two"], [None], [None], [None], ["three"]], transformer.transform(X).tolist())

	def test_transform_object(self):
		mapping = {
			tuple(["zero"]) : "null",
			("one", True) : "ein",
			("two", True) : "zwei",
			("three", True) : "drei"
		}
		with self.assertRaises(TypeError):
			MultiLookupTransformer(mapping, None)
		mapping[("zero", int(0))] = mapping.pop(tuple(["zero"]))
		with self.assertRaises(TypeError):
			MultiLookupTransformer(mapping, None)
		mapping.pop(("zero", int(0)))
		transformer = MultiLookupTransformer(mapping, None)
		X = DataFrame([["one", None], ["one", True], [None, True], ["two", True], ["three", True]])
		self.assertEqual([[None], ["ein"], [None], ["zwei"], ["drei"]], transformer.transform(X).tolist())
		X = numpy.array([["one", True], ["one", None], ["one", False], ["two", True]], dtype = "O")
		self.assertEqual([["ein"], [None], [None], ["zwei"]], transformer.transform(X).tolist())
		transformer = MultiLookupTransformer(mapping, "(other)")
		self.assertEqual([["ein"], [None], ["(other)"], ["zwei"]], transformer.transform(X).tolist())

	def test_transform_categorical(self):
		X = DataFrame([["apple", "green"], ["banana", "green"], ["banana", "yellow"]], columns = ["fruit", "color"])
		mapping = {
			("apple", "red") : True,
			("banana", "green") : False,
			("banana", "yellow") : True
		}
		transformer = MultiLookupTransformer(mapping, None)
		Xt = transformer.fit_transform(X)
		self.assertIsInstance(Xt, numpy.ndarray)
		self.assertEqual(object, Xt.dtype)
		self.assertTrue(_list_equal([float("NaN"), False, True], Xt.tolist()))
		transformer = MultiLookupTransformer(mapping, None, dtype = "category")
		Xt = transformer.fit_transform(X)
		self.assertIsInstance(Xt, Series)
		self.assertIsInstance(Xt.dtype, CategoricalDtype)
		self.assertEqual([False, True], Xt.dtype.categories.tolist())
		self.assertTrue(_list_equal([float("NaN"), False, True], Xt.tolist()))

class PMMLLabelBinarizerTest(TestCase):

	def test_fit_float(self):
		X = [1.0, float("NaN"), 1.0, 2.0, float("NaN"), 3.0, 3.0, 2.0]
		labels = [1.0, 2.0, 3.0]
		binarizer = PMMLLabelBinarizer()
		binarizer.fit(X)
		self.assertEqual(labels, binarizer.classes_.tolist())

	def test_fit_string(self):
		X = ["A", None, "A", "B", None, "C", "C", "B"]
		labels = ["A", "B", "C"]
		binarizer = PMMLLabelBinarizer()
		self.assertFalse(hasattr(binarizer, "classes_"))
		binarizer.fit(X)
		self.assertEqual(labels, binarizer.classes_.tolist())
		binarizer.fit(numpy.array(X))
		self.assertEqual(labels, binarizer.classes_.tolist())
		binarizer.fit(Series(numpy.array(X)))
		self.assertEqual(labels, binarizer.classes_.tolist())

	def test_transform_float(self):
		X = [1.0, 2.0, 3.0]
		dense_binarizer = PMMLLabelBinarizer()
		dense_binarizer.fit(X)
		Xt_dense = dense_binarizer.transform([1.0, 3.0, float("NaN"), -1.0, 2.0])
		self.assertIsInstance(Xt_dense, numpy.ndarray)
		self.assertEqual([[1, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 1, 0]], Xt_dense.tolist())
		sparse_binarizer = PMMLLabelBinarizer(sparse_output = True)
		sparse_binarizer.fit(X)
		Xt_sparse = sparse_binarizer.transform([1.0, 3.0, float("NaN"), -1.0, 2.0])
		self.assertIsInstance(Xt_sparse, scipy.sparse.csr_matrix)
		self.assertEqual(Xt_dense.tolist(), Xt_sparse.toarray().tolist())

	def test_transform_string(self):
		X = ["A", "B", "C"]
		dense_binarizer = PMMLLabelBinarizer()
		dense_binarizer.fit(X)
		Xt_dense = dense_binarizer.transform(["A", "C", None, "D", "B"])
		self.assertIsInstance(Xt_dense, numpy.ndarray)
		self.assertEqual([[1, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 1, 0]], Xt_dense.tolist())
		sparse_binarizer = PMMLLabelBinarizer(sparse_output = True)
		sparse_binarizer.fit(X)
		Xt_sparse = sparse_binarizer.transform(["A", "C", None, "D", "B"])
		self.assertIsInstance(Xt_sparse, scipy.sparse.csr_matrix)
		self.assertEqual(Xt_dense.tolist(), Xt_sparse.toarray().tolist())

class PMMLLabelEncoderTest(TestCase):

	def test_fit_float(self):
		X = [1.0, float("NaN"), 1.0, 2.0, float("NaN"), 3.0, 3.0, 2.0]
		labels = [1.0, 2.0, 3.0]
		encoder = PMMLLabelEncoder(missing_values = -999)
		self.assertEqual(-999, encoder.missing_values)
		encoder.fit(X)
		self.assertEqual(labels, encoder.classes_.tolist())

	def test_fit_string(self):
		X = ["A", None, "A", "B", None, "C", "C", "B"]
		labels = ["A", "B", "C"]
		encoder = PMMLLabelEncoder()
		self.assertFalse(hasattr(encoder, "classes_"))
		encoder.fit(X)
		self.assertEqual(labels, encoder.classes_.tolist())
		encoder.fit(numpy.array(X))
		self.assertEqual(labels, encoder.classes_.tolist())
		encoder.fit(Series(numpy.array(X)))
		self.assertEqual(labels, encoder.classes_.tolist())

	def test_transform_float(self):
		X = [1.0, 2.0, 3.0]
		encoder = PMMLLabelEncoder(missing_values = -999)
		encoder.fit(X)
		self.assertEqual([[0], [2], [-999], [-999], [1]], encoder.transform([1.0, 3.0, float("NaN"), -1.0, 2.0]).tolist())

	def test_transform_string(self):
		X = ["A", "B", "C"]
		encoder = PMMLLabelEncoder()
		encoder.fit(X)
		self.assertEqual([[0], [2], [None], [None], [1]], encoder.transform(["A", "C", None, "D", "B"]).tolist())
		self.assertEqual([[None]], encoder.transform(numpy.array([None])).tolist())
		self.assertEqual([[0], [1], [2]], encoder.transform(Series(numpy.array(["A", "B", "C"]))).tolist())

class PowerFunctionTransformerTest(TestCase):

	def test_transform(self):
		X = numpy.asarray([-2, -1, 0, 1, 2])
		transformer = PowerFunctionTransformer(power = 1)
		self.assertEqual(X.tolist(), transformer.transform(X).tolist())
		transformer = PowerFunctionTransformer(power = 2)
		self.assertEqual([4, 1, 0, 1, 4], transformer.transform(X).tolist())
		transformer = PowerFunctionTransformer(power = 3)
		self.assertEqual([-8, -1, 0, 1, 8], transformer.transform(X).tolist())

class LagTransformerTest(TestCase):

	def test_transform(self):
		X = Series([-1, 0, 1, 2, 3])
		transformer = LagTransformer(n = 2)
		Xt = transformer.transform(X)
		self.assertIsInstance(Xt, Series)
		self.assertNotEqual(X.dtype, Xt.dtype)
		self.assertTrue(_list_equal([float("NaN"), float("NaN"), -1.0, 0.0, 1.0], Xt.tolist()))
		X = Series(["-1", "0", "1", "2", "3"])
		Xt = transformer.transform(X)
		self.assertIsInstance(Xt, Series)
		self.assertEqual(X.dtype, Xt.dtype)
		self.assertEqual([None, None, "-1", "0", "1"], Xt.tolist())

		X = DataFrame([[-1.0, -1, False, "-1"], [0.0, 0, False, "0"], [1.0, 1, True, "1"], [2.0, 2, True, "2"], [3.0, 3, True, "3"]], columns = ["float", "int", "bool", "str"])
		Xt = transformer.transform(X)
		self.assertIsInstance(Xt, DataFrame)
		self.assertEqual(X.shape, Xt.shape)
		self.assertTrue(_list_equal([float("NaN"), float("NaN"), -1.0, 0.0, 1.0], Xt["float"].tolist()))
		self.assertTrue(_list_equal([float("NaN"), float("NaN"), -1.0, 0.0, 1.0], Xt["int"].tolist()))
		self.assertTrue(_list_equal([float("NaN"), float("NaN"), False, False, True], Xt["bool"].tolist()))
		self.assertTrue(_list_equal([float("NaN"), float("NaN"), "-1", "0", "1"], Xt["str"].tolist()))
		X = X.values
		Xt = transformer.transform(X)
		self.assertIsInstance(Xt, numpy.ndarray)
		self.assertEqual(X.shape, Xt.shape)
		self.assertTrue(_list_equal([float("NaN"), float("NaN"), float("NaN"), float("NaN")], Xt[0, :].tolist()))
		self.assertTrue(_list_equal([float("NaN"), float("NaN"), float("NaN"), float("NaN")], Xt[1, :].tolist()))
		self.assertEqual([-1.0, -1, False, "-1"], Xt[2, :].tolist())
		self.assertEqual([0.0, 0, False, "0"], Xt[3, :].tolist())
		self.assertEqual([1.0, 1, True, "1"], Xt[4, :].tolist())

		X = DataFrame([[-1, "a"], [0, "a"], [1, "b"], [2, "b"], [3, "a"]], columns = ["int", "block"])
		transformer = LagTransformer(n = 1, block_indicators = ["block"])
		Xt = transformer.transform(X)
		self.assertIsInstance(Xt, DataFrame)
		self.assertEqual(X.shape, Xt.shape)
		self.assertTrue(_list_equal([float("NaN"), -1, float("NaN"), 1, 0], Xt["int"].tolist()))
		self.assertEqual(["a", "a", "b", "b", "a"], Xt["block"].tolist())
		X = X.values
		transformer = LagTransformer(n = 1, block_indicators = [1])
		Xt = transformer.transform(X)
		self.assertIsInstance(Xt, numpy.ndarray)
		self.assertEqual(X.shape, Xt.shape)
		self.assertTrue(_list_equal([float("NaN"), -1, float("NaN"), 1, 0], Xt[:, 0].tolist()))
		self.assertEqual(["a", "a", "b", "b", "a"], Xt[:, 1].tolist())

	def test_get_feature_names_out(self):
		transformer = LagTransformer(n = 1, block_indicators = [0])
		self.assertEqual(["x1", "x2_lag1", "x3_lag1"], transformer.get_feature_names_out(["x1", "x2", "x3"]).tolist())
		transformer = LagTransformer(n = 2, block_indicators = ["x3"])
		self.assertEqual(["x1_lag2", "x2_lag2", "x3"], transformer.get_feature_names_out(["x1", "x2", "x3"]).tolist())

class RollingAggregateTransformerTest(TestCase):

	def _transform1d(self, transformer, X):
		self.assertIsInstance(X, Series)
		X_ndarray = X.values.reshape(-1, 1)
		self.assertIsInstance(X_ndarray, numpy.ndarray)
		Xt = transformer.transform(X)
		Xt_ndarray = transformer.transform(X_ndarray).ravel()
		self.assertIsInstance(Xt, Series)
		self.assertIsInstance(Xt_ndarray, numpy.ndarray)
		self.assertTrue(_list_equal(Xt.tolist(), Xt_ndarray.tolist()))
		return Xt

	def _transform2d(self, transformer, X):
		self.assertIsInstance(X, DataFrame)
		X_ndarray = X.values
		self.assertIsInstance(X_ndarray, numpy.ndarray)
		Xt = transformer.transform(X)
		Xt_ndarray = transformer.transform(X_ndarray)
		self.assertIsInstance(Xt, DataFrame)
		self.assertIsInstance(Xt_ndarray, numpy.ndarray)
		self.assertTrue(_list_equal(Xt.values.tolist(), Xt_ndarray.tolist()))
		return Xt

	def test_avg(self):
		X = Series([1, 2, 3, 4, 5])
		transformer = RollingAggregateTransformer(function = "avg", n = 2)
		Xt = self._transform1d(transformer, X)
		self.assertTrue(_list_equal([float("NaN"), 1.0, 1.5, 2.5, 3.5], Xt.tolist()))

		X = DataFrame([[1, -1], [2, numpy.nan], [3, 1], [4, 2], [5, 3]])
		Xt = self._transform2d(transformer, X)
		self.assertTrue(_list_equal([[None, None], [1, -1], [1.5, -1], [2.5, 1], [3.5, 1.5]], Xt.values.tolist()))

	def test_max(self):
		X = Series([1, 2, 3, 4, 5])
		transformer = RollingAggregateTransformer(function = "max", n = 3)
		Xt = self._transform1d(transformer, X)
		self.assertTrue(_list_equal([float("NaN"), 1, 2, 3, 4], Xt.tolist()))

		X = DataFrame([[1, -1], [2, numpy.nan], [3, 1], [4, 2], [5, 3]])
		Xt = self._transform2d(transformer, X)
		self.assertTrue(_list_equal([[None, None], [1, -1], [2, -1], [3, 1], [4, 2]], Xt.values.tolist()))

	def test_min(self):
		X = Series([1, 2, 3, 4, 5])
		transformer = RollingAggregateTransformer(function = "min", n = 3)
		Xt = self._transform1d(transformer, X)
		self.assertTrue(_list_equal([float("NaN"), 1, 1, 1, 2], Xt.tolist()))

		X = DataFrame([[1, -1], [2, numpy.nan], [3, 1], [4, 2], [5, 3]])
		Xt = self._transform2d(transformer, X)
		self.assertTrue(_list_equal([[None, None], [1, -1], [1, -1], [1, -1], [2, 1]], Xt.values.tolist()))

	def test_product(self):
		X = Series([1, 2, 3, 4, 5])
		transformer = RollingAggregateTransformer(function = "product", n = 3)
		Xt = self._transform1d(transformer, X)
		self.assertTrue(_list_equal([float("NaN"), 1, 2, 6, 24], Xt.tolist()))

		X = DataFrame([[1, -1], [2, numpy.nan], [3, 1], [4, 2], [5, 3]])
		Xt = self._transform2d(transformer, X)
		self.assertTrue(_list_equal([[float("NaN"), float("NaN")], [1, -1], [2, -1], [6, -1], [24, 2]], Xt.values.tolist()))

	def test_sum(self):
		X = Series([1, 2, 3, 4, 5])
		transformer = RollingAggregateTransformer(function = "sum", n = 3)
		Xt = self._transform1d(transformer, X)
		self.assertTrue(_list_equal([float("NaN"), 1.0, 3.0, 6.0, 9.0], Xt.tolist()))

		X = DataFrame([[1, -1], [2, numpy.nan], [3, 1], [4, 2], [5, 3]])
		Xt = self._transform2d(transformer, X)
		self.assertTrue(_list_equal([[float("NaN"), float("NaN")], [1, -1], [3, -1], [6, 0], [9, 3]], Xt.values.tolist()))

		X = DataFrame([[-1, "a"], [0, "a"], [1, "b"], [2, "b"], [3, "a"], [4, "b"], [5, "a"]], columns = ["int", "block"])
		transformer = RollingAggregateTransformer(function = "sum", n = 2, block_indicators = ["block"])
		Xt = transformer.fit_transform(X)
		self.assertTrue(_list_equal([float("NaN"), -1, float("NaN"), 1, -1, 3, 3], Xt.iloc[:, 0].tolist()))
		X = X.values
		transformer = RollingAggregateTransformer(function = "sum", n = 2, block_indicators = [1])
		Xt = transformer.fit_transform(X)
		self.assertTrue(_list_equal([float("NaN"), -1, float("NaN"), 1, -1, 3, 3], Xt[:, 0].tolist()))

	def test_get_feature_names_out(self):
		transformer = RollingAggregateTransformer(function = "avg", n = 100, block_indicators = [0, 1])
		self.assertEqual(["x1", "x2", "x3_avg100"], transformer.get_feature_names_out(["x1", "x2", "x3"]).tolist())

class ConcatTransformerTest(TestCase):

	def test_transform(self):
		X = numpy.asarray([["A", 1, "C"], [1, 2, 3], ["x", "y", "z"]])
		transformer = ConcatTransformer()
		self.assertEqual([["A1C"], ["123"], ["xyz"]], transformer.transform(X).tolist())
		X = DataFrame([["L", -1], ["R", 1]], columns = ["left", "right"])
		transformer = ConcatTransformer("/")
		self.assertEqual([["L/-1"], ["R/1"]], transformer.transform(X).tolist())

class MatchesTransformerTest(TransformerTest):

	def test_transform(self):
		X = Series(["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])
		transformer = MatchesTransformer("ar?y")
		self.assertEqual([True, True, False, False, True, False, False, False, False, False, False, False], self._transform1d(transformer, X).tolist())

class ReplaceTransformerTest(TransformerTest):

	def test_transform(self):
		X = Series(["A", "B", "BA", "BB", "BAB", "ABBA", "BBBB"])
		transformer = ReplaceTransformer("B+", "c")
		self.assertEqual(["A", "c", "cA", "c", "cAc", "AcA", "c"], self._transform1d(transformer, X).tolist())
		vectorizer = CountVectorizer(token_pattern = r"\w+")
		pipeline = make_pipeline(transformer, vectorizer)
		Xt = pipeline.fit_transform(X)
		self.assertEqual((7, 5), Xt.shape)
		X = X.values
		Xt = pipeline.fit_transform(X)
		self.assertEqual((7, 5), Xt.shape)

class StringLengthTransformerTest(TransformerTest):

	def test_transform(self):
		X = Series(["one", "two", "three", "four", "five"])
		transformer = StringLengthTransformer()
		self.assertEqual([3, 3, 5, 4, 4], self._transform1d(transformer, X).tolist())

class StringNormalizerTest(TransformerTest):

	def test_transform(self):
		X = Series([" One", " two ", "THRee "])
		transformer = StringNormalizer(function = None)
		self.assertEqual(["One", "two", "THRee"], self._transform1d(transformer, X).tolist())
		transformer = StringNormalizer(function = "uppercase", trim_blanks = False)
		self.assertEqual([" ONE", " TWO ", "THREE "], self._transform1d(transformer, X).tolist())
		transformer = StringNormalizer(function = "lowercase")
		self.assertEqual(["one", "two", "three"], self._transform1d(transformer, X).tolist())
		vectorizer = CountVectorizer()
		pipeline = make_pipeline(transformer, vectorizer)
		Xt = pipeline.fit_transform(X)
		self.assertEqual((3, 3), Xt.shape)
		X = X.values
		Xt = pipeline.fit_transform(X)
		self.assertEqual((3, 3), Xt.shape)

		X = X.reshape((3, 1))
		pipeline = make_pipeline(StringNormalizer(), OrdinalEncoder())
		Xt = pipeline.fit_transform(X)
		self.assertEqual((3, 1), Xt.shape)

class SubstringTransformerTest(TransformerTest):

	def test_transform(self):
		X = Series(["", "a", "aB", "aBc", "aBc9", "aBc9x"])
		transformer = SubstringTransformer(1, 4)
		self.assertEqual(["", "", "B", "Bc", "Bc9", "Bc9"], self._transform1d(transformer, X).tolist())
		vectorizer = CountVectorizer(token_pattern = r"\w+")
		pipeline = make_pipeline(transformer, vectorizer)
		Xt = pipeline.fit_transform(X)
		self.assertEqual((6, 3), Xt.shape)
		X = X.values
		Xt = pipeline.fit_transform(X)
		self.assertEqual((6, 3), Xt.shape)

class WordCountTransformerTest(TransformerTest):

	def test_transform(self):
		X = Series(["", "Hellow World", "Happy New Year", "!?"])
		transformer = WordCountTransformer()
		self.assertEqual([[0], [2], [3], [0]], transformer.transform(X).tolist())
		X = X.values
		self.assertEqual([[0], [2], [3], [0]], transformer.transform(X).tolist())

class SelectFirstTransformerTest(TestCase):

	def test_fit_transform(self):
		X = DataFrame([["A", 1.0], ["B", 0], ["A", 3.0], ["C", -1.5]], columns = ["subset", "value"])
		transformer = SelectFirstTransformer([
			("A", ExpressionTransformer("X['value'] + 1.0"), "X['subset'] == 'A'"),
			("B", ExpressionTransformer("X['value'] - 1.0"), "X['subset'] not in ['A', 'C']")
		], eval_rows = True)
		Xt = transformer.fit_transform(X)
		self.assertEqual([[2.0], [-1.0], [4.0], [None]], Xt.tolist())
		X = X.values
		transformer = SelectFirstTransformer([
			("A", ExpressionTransformer("X[1] + 1.0"), "X[:, 0] == 'A'"),
			("B", ExpressionTransformer("X[1] - 1.0"), "numpy.logical_and(X[:, 0] != 'A', X[:, 0] != 'C')")
		], eval_rows = False)
		Xt = transformer.fit_transform(X)
		self.assertEqual([[2.0], [-1.0], [4.0], [None]], Xt.tolist())

class H2OFrameCreatorTest(TestCase):

	def test_init(self):
		with self.assertWarns(DeprecationWarning):
			H2OFrameCreator()

def _data():
	strings = Series(["0", "1", "2"], name = "string", dtype = str)
	booleans = Series([False, True, True], name = "boolean", dtype = bool)
	cont_ints = Series([0, 1, 2], name = "cont_int", dtype = int)
	cat_ints = Series([0, 1, 2], name = "cat_int", dtype = int).astype("category")
	floats = Series([0.0, 1.0, 2.0], name = "float", dtype = float)
	return pandas.concat([strings, booleans, cont_ints, cat_ints, floats], axis = 1)

class LightGBMTest(TestCase):

	def test_fit_transform(self):
		X = _data()
		ct, ct_categorical_feature = make_lightgbm_column_transformer(X.dtypes, missing_value_aware = False)
		dfm, dfm_categorical_feature = make_lightgbm_dataframe_mapper(X.dtypes, missing_value_aware = False)
		self.assertEqual(ct.fit_transform(X).tolist(), dfm.fit_transform(X).tolist())
		self.assertEqual([0, 1, 3], ct_categorical_feature)
		self.assertEqual([0, 1, 3], dfm_categorical_feature)
		ct, ct_categorical_feature = make_lightgbm_column_transformer(X.dtypes, missing_value_aware = True)
		dfm, dfm_categorical_feature = make_lightgbm_dataframe_mapper(X.dtypes, missing_value_aware = True)
		self.assertEqual(ct.fit_transform(X).tolist(), dfm.fit_transform(X).tolist())
		self.assertEqual([0, 1, 3], ct_categorical_feature)
		self.assertEqual([0, 1, 3], dfm_categorical_feature)

class XGBoostTest(TestCase):

	def test_fit_transform(self):
		X = _data()
		ct = make_xgboost_column_transformer(X.dtypes, missing_value_aware = False)
		dfm = make_xgboost_column_transformer(X.dtypes, missing_value_aware = False)
		self.assertEqual(ct.fit_transform(X).tolist(), dfm.fit_transform(X).tolist())
		ct = make_xgboost_column_transformer(X.dtypes, missing_value_aware = True)
		dfm = make_xgboost_column_transformer(X.dtypes, missing_value_aware = True)
		self.assertEqual(ct.fit_transform(X).tolist(), dfm.fit_transform(X).tolist())
