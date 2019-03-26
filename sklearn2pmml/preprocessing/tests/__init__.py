from datetime import datetime
from pandas import DataFrame, Series
from sklearn.preprocessing import Imputer
from sklearn_pandas import DataFrameMapper
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn2pmml.decoration import Alias, DateDomain, DateTimeDomain
from sklearn2pmml.preprocessing import Aggregator, ConcatTransformer, CutTransformer, DaysSinceYearTransformer, ExpressionTransformer, LookupTransformer, MatchesTransformer, MultiLookupTransformer, PMMLLabelBinarizer, PMMLLabelEncoder, PowerFunctionTransformer, ReplaceTransformer, SecondsSinceYearTransformer, StringNormalizer, SubstringTransformer
from unittest import TestCase

import math
import numpy

class AggregatorTest(TestCase): 

	def test_transform(self):
		X = numpy.asarray([1, 0.5, 2, 3.0, 0, 1.0])
		aggregator = Aggregator(function = "min")
		X = X.reshape((1, 6))
		self.assertEqual(0, aggregator.transform(X))
		X = X.reshape((3, 2))
		self.assertEqual([0.5, 2, 0], aggregator.transform(X).tolist())
		X = X.reshape((2, 3))
		self.assertEqual([0.5, 0], aggregator.transform(X).tolist())
		X = X.reshape((6, 1))
		self.assertEqual([1, 0.5, 2, 3.0, 0, 1.0], aggregator.transform(X).tolist())

class CutTransformerTest(TestCase):

	def test_transform(self):
		bins = [float("-inf"), -1.0, 0.0, 1.0, float("+inf")]
		transformer = CutTransformer(bins, labels = False, right = True)
		X = numpy.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
		self.assertEqual([0, 0, 1, 1, 2, 2, 3], transformer.transform(X).tolist())
		transformer = CutTransformer(bins, labels = False, right = False)
		self.assertEqual([0, 1, 1, 2, 2, 3, 3], transformer.transform(X).tolist())
		bins = [-3.0, -1.0, 1.0, 3.0]
		transformer = CutTransformer(bins, labels = False, right = True, include_lowest = True)
		X = numpy.array([-3.0, -2.0, 2.0, 3.0])
		self.assertEqual([0, 0, 2, 2], transformer.transform(X).tolist())
		X = numpy.array([-5.0])
		self.assertTrue(numpy.isnan(transformer.transform(X)).tolist()[0])
		X = numpy.array([5.0])
		self.assertTrue(numpy.isnan(transformer.transform(X)).tolist()[0])

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
		pipeline = Pipeline([
			("union", FeatureUnion([
				("left_mapper", DataFrameMapper([
					("left", [DateDomain(), DaysSinceYearTransformer(year = 2010)])
				])),
				("right_mapper", DataFrameMapper([
					("right", [DateDomain(), DaysSinceYearTransformer(year = 2010)])
				]))
			])),
			("expression", Alias(ExpressionTransformer("X[0] - X[1]"), "delta(left, right)", prefit = True))
		])
		Xt = pipeline.fit_transform(X)
		self.assertEqual([[-1], [30]], Xt.tolist())

	def test_timedelta_seconds(self):
		X = DataFrame([["2018-12-31T23:59:59", "2019-01-01T00:00:00"], ["2019-01-01T03:30:03", "2019-01-01T00:00:00"]], columns = ["left", "right"])
		mapper = DataFrameMapper([
			(["left", "right"], [DateTimeDomain(), SecondsSinceYearTransformer(year = 2010), ExpressionTransformer("X[0] - X[1]")])
		])
		Xt = mapper.fit_transform(X)
		self.assertEqual([[-1], [12603]], Xt.tolist())

class ExpressionTransformerTest(TestCase):

	def test_transform(self):
		transformer = ExpressionTransformer("X['a'] + X['b']")
		X = DataFrame([[0.5, 0.5], [1.0, 2.0]], columns = ["a", "b"])
		Xt = transformer.fit_transform(X)
		self.assertIsInstance(Xt, numpy.ndarray)
		self.assertEqual([[1.0], [3.0]], Xt.tolist())
		transformer = ExpressionTransformer("X[0] + X[1]")
		self.assertTrue(hasattr(transformer, "expr"))
		X = numpy.array([[0.5, 0.5], [1.0, 2.0]])
		Xt = transformer.fit_transform(X)
		self.assertIsInstance(Xt, numpy.ndarray)
		self.assertEqual([[1.0], [3.0]], Xt.tolist())
		transformer = ExpressionTransformer("X[0] - X[1]")
		self.assertEqual([[0.0], [-1.0]], transformer.fit_transform(X).tolist())
		transformer = ExpressionTransformer("X[0] * X[1]")
		self.assertEqual([[0.25], [2.0]], transformer.fit_transform(X).tolist())
		transformer = ExpressionTransformer("X[0] / X[1]")
		self.assertEqual([[1.0], [0.5]], transformer.fit_transform(X).tolist())

	def test_sequence_transform(self):
		X = DataFrame([[None], [1], [None]], columns = ["a"])
		mapper = DataFrameMapper([
			(["a"], [ExpressionTransformer("0 if pandas.isnull(X[0]) else X[0]"), Imputer(missing_values = 0)])
		])
		Xt = mapper.fit_transform(X)
		self.assertEqual([[1], [1], [1]], Xt.tolist())

class LookupTransformerTest(TestCase):

	def test_transform_float(self):
		mapping = {
			0.0 : math.cos(0.0),
			45.0 : math.cos(45.0),
			90.0 : math.cos(90.0)
		}
		transformer = LookupTransformer(mapping, float("NaN"))
		self.assertEqual([math.cos(0.0), math.cos(90.0)], transformer.transform([0.0, 90.0]).tolist())
		self.assertTrue(math.isnan(transformer.transform([180.0])))
		self.assertEqual([math.cos(0.0), math.cos(45.0), math.cos(90.0)], transformer.transform(Series(numpy.array([0.0, 45.0, 90.0]))).tolist())

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
		self.assertEqual([None, "ein"], transformer.transform(["zero", "one"]).tolist())
		self.assertEqual(["ein", "zwei", "drei"], transformer.transform(Series(numpy.array(["one", "two", "three"]))).tolist())

class MultiLookupTransformerTest(TestCase):

	def test_transform_int(self):
		mapping = {
			(1, 1) : "one",
			(2, 2) : "two",
			(3, 3) : "three"
		}
		transformer = MultiLookupTransformer(mapping, None)
		X = DataFrame([[1, 0], [1, 1], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1], [3, 2], [3, 3]])
		self.assertEqual([None, "one", None, None, "two", None, None, None, "three"], transformer.transform(X).tolist())

	def test_transform_object(self):
		mapping = {
			tuple(["zero"]) : "null",
			("one", True) : "ein",
			("two", True) : "zwei",
			("three", True) : "drei"
		}
		with self.assertRaises(ValueError):
			MultiLookupTransformer(mapping, None)
		mapping.pop(tuple(["zero"]))
		transformer = MultiLookupTransformer(mapping, None)
		X = DataFrame([["one", None], ["one", True], [None, True], ["two", True], ["three", True]])
		self.assertEqual([None, "ein", None, "zwei", "drei"], transformer.transform(X).tolist())
		X = numpy.matrix([["one", True], ["one", None], ["two", True]], dtype = "O")
		self.assertEqual(["ein", None, "zwei"], transformer.transform(X).tolist())

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
		X = [1.0, float("NaN"), 2.0, 3.0]
		binarizer = PMMLLabelBinarizer()
		binarizer.fit(X)
		self.assertEqual([[1, 0, 0], [0, 0, 1], [0, 0, 0], [0, 1, 0]], binarizer.transform([1.0, 3.0, float("NaN"), 2.0]).tolist())

	def test_transform_string(self):
		X = ["A", None, "B", "C"]
		binarizer = PMMLLabelBinarizer()
		binarizer.fit(X)
		self.assertEqual([[1, 0, 0], [0, 0, 1], [0, 0, 0], [0, 1, 0]], binarizer.transform(["A", "C", None, "B"]).tolist())
		self.assertEqual([[0, 0, 0]], binarizer.transform([None]).tolist())
		self.assertEqual([[1, 0, 0], [0, 1, 0], [0, 0, 1]], binarizer.transform(["A", "B", "C"]).tolist())

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
		X = [1.0, float("NaN"), 2.0, 3.0]
		encoder = PMMLLabelEncoder(missing_values = -999)
		encoder.fit(X)
		self.assertEqual([0, 2, -999, 1], encoder.transform([1.0, 3.0, float("NaN"), 2.0]).tolist())

	def test_transform_string(self):
		X = ["A", None, "B", "C"]
		encoder = PMMLLabelEncoder()
		encoder.fit(X)
		self.assertEqual([0, 2, None, 1], encoder.transform(["A", "C", None, "B"]).tolist())
		self.assertEqual([None], encoder.transform(numpy.array([None])).tolist())
		self.assertEqual([0, 1, 2], encoder.transform(Series(numpy.array(["A", "B", "C"]))).tolist())

class PowerFunctionTransformerTest(TestCase):

	def test_transform(self):
		X = numpy.asarray([-2, -1, 0, 1, 2])
		transformer = PowerFunctionTransformer(power = 1)
		self.assertEqual(X.tolist(), transformer.transform(X).tolist())
		transformer = PowerFunctionTransformer(power = 2)
		self.assertEqual([4, 1, 0, 1, 4], transformer.transform(X).tolist())
		transformer = PowerFunctionTransformer(power = 3)
		self.assertEqual([-8, -1, 0, 1, 8], transformer.transform(X).tolist())

class ConcatTransformerTest(TestCase):

	def test_transform(self):
		X = numpy.asarray([["A", 1, "C"], [1, 2, 3], ["x", "y", "z"]])
		transformer = ConcatTransformer()
		self.assertEqual([["A1C"], ["123"], ["xyz"]], transformer.transform(X).tolist())
		X = DataFrame([["L", -1], ["R", 1]], columns = ["left", "right"])
		transformer = ConcatTransformer("/")
		self.assertEqual([["L/-1"], ["R/1"]], transformer.transform(X).tolist())

class MatchesTransformerTest(TestCase):

	def test_transform(self):
		X = numpy.asarray(["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])
		Xt_exp = [True, True, False, False, True, False, False, False, False, False, False, False]
		transformer = MatchesTransformer("ar?y")
		self.assertEqual(Xt_exp, transformer.transform(X).tolist())
		X = DataFrame(X.reshape(-1, 1), columns = ["month"])
		Xt = transformer.transform(X)
		self.assertTrue((12, 1), Xt.shape)
		self.assertEqual(Xt_exp, Xt.tolist())

class ReplaceTransformerTest(TestCase):

	def test_transform(self):
		X = numpy.asarray(["A", "B", "BA", "BB", "BAB", "ABBA", "BBBB"])
		Xt_exp = ["A", "c", "cA", "c", "cAc", "AcA", "c"]
		transformer = ReplaceTransformer("B+", "c")
		self.assertEqual(Xt_exp, transformer.transform(X).tolist())
		X = DataFrame(X.reshape(-1, 1), columns = ["input"])
		Xt = transformer.transform(X)
		self.assertTrue((7, 1), Xt.shape)
		self.assertEqual(Xt_exp, Xt.tolist())

class StringNormalizerTest(TestCase):

	def test_transform(self):
		X = numpy.asarray([" One", " two ", "THRee "])
		normalizer = StringNormalizer(function = None)
		self.assertEqual(["One", "two", "THRee"], normalizer.transform(X).tolist())
		normalizer = StringNormalizer(function = "uppercase", trim_blanks = False)
		self.assertEqual([" ONE", " TWO ", "THREE "], normalizer.transform(X).tolist())
		normalizer = StringNormalizer(function = "lowercase")
		self.assertEqual(["one", "two", "three"], normalizer.transform(X).tolist())
		X = Series(X, dtype = str)
		self.assertEqual(["one", "two", "three"], normalizer.transform(X).tolist())

class SubstringTransformerTest(TestCase):

	def test_transform(self):
		X = numpy.asarray(["", "a", "aB", "aBc", "aBc9", "aBc9x"])
		transformer = SubstringTransformer(1, 4)
		self.assertEqual(["", "", "B", "Bc", "Bc9", "Bc9"], transformer.transform(X).tolist())
		X = DataFrame(X, columns = ["input"])
		self.assertEqual(["", "", "B", "Bc", "Bc9", "Bc9"], transformer.transform(X).tolist())