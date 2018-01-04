from pandas import DataFrame, Series
from sklearn2pmml.preprocessing import ExpressionTransformer, LookupTransformer, MultiLookupTransformer, PMMLLabelBinarizer, PMMLLabelEncoder
from unittest import TestCase

import math
import numpy

class ExpressionTransformerTest(TestCase):

	def test_transform(self):
		transformer = ExpressionTransformer("X[:, 0] + X[:, 1]")
		self.assertTrue(hasattr(transformer, "expr_"))
		X = numpy.array([[0.5, 0.5], [1.0, 2.0]])
		Xt = transformer.fit_transform(X)
		self.assertEqual([1.0, 3.0], Xt.tolist())
		transformer = ExpressionTransformer("X[:, 0] - X[:, 1]")
		Xt = transformer.fit_transform(X)
		self.assertEqual([0.0, -1.0], Xt.tolist())
		transformer = ExpressionTransformer("X[:, 0] * X[:, 1]")
		Xt = transformer.fit_transform(X)
		self.assertEqual([0.25, 2.0], Xt.tolist())
		transformer = ExpressionTransformer("X[:, 0] / X[:, 1]")
		Xt = transformer.fit_transform(X)
		self.assertEqual([1.0, 0.5], Xt.tolist())

class LookupTransformerTest(TestCase):

	def test_transform_float(self):
		mapping = {0.0 : math.cos(0.0), 45.0 : math.cos(45.0), 90.0 : math.cos(90.0)}
		transformer = LookupTransformer(mapping, float("NaN"))
		self.assertEqual([math.cos(0.0), math.cos(90.0)], transformer.transform([0.0, 90.0]).tolist())
		self.assertTrue(math.isnan(transformer.transform([180.0])))
		self.assertEqual([math.cos(0.0), math.cos(45.0), math.cos(90.0)], transformer.transform(Series(numpy.array([0.0, 45.0, 90.0]))).tolist())

	def test_transform_string(self):
		mapping = {None : "null", "one" : "ein", "two" : "zwei", "three" : "drei"}
		with self.assertRaises(ValueError):
			LookupTransformer(mapping, None)
		mapping.pop(None)
		transformer = LookupTransformer(mapping, None)
		self.assertEqual([None, "ein"], transformer.transform(["zero", "one"]).tolist())
		self.assertEqual(["ein", "zwei", "drei"], transformer.transform(Series(numpy.array(["one", "two", "three"]))).tolist())

class MultiLookupTransformerTest(TestCase):

	def test_transform_int(self):
		mapping = {(1, 1) : "one", (2, 2) : "two", (3, 3) : "three"}
		transformer = MultiLookupTransformer(mapping, None)
		Y = DataFrame([[1, 0], [1, 1], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1], [3, 2], [3, 3]])
		Yt = transformer.transform(Y)
		self.assertEqual([None, "one", None, None, "two", None, None, None, "three"], Yt.tolist())

	def test_transform_object(self):
		mapping = {tuple(["zero"]) : "null", ("one", True) : "ein", ("two", True) : "zwei", ("three", True) : "drei"}
		with self.assertRaises(ValueError):
			MultiLookupTransformer(mapping, None)
		mapping.pop(tuple(["zero"]))
		transformer = MultiLookupTransformer(mapping, None)
		Y = DataFrame([["one", None], ["one", True], [None, True], ["two", True], ["three", True]])
		Yt = transformer.transform(Y)
		self.assertEqual([None, "ein", None, "zwei", "drei"], Yt.tolist())
		Y = numpy.matrix([["one", True], ["one", None], ["two", True]], dtype = "O")
		Yt = transformer.transform(Y)
		self.assertEqual(["ein", None, "zwei"], Yt.tolist())

class PMMLLabelBinarizerTest(TestCase):

	def test_fit_float(self):
		y = [1.0, float("NaN"), 1.0, 2.0, float("NaN"), 3.0, 3.0, 2.0]
		labels = [1.0, 2.0, 3.0]
		binarizer = PMMLLabelBinarizer()
		binarizer.fit(y)
		self.assertEqual(labels, binarizer.classes_.tolist())

	def test_fit_string(self):
		y = ["A", None, "A", "B", None, "C", "C", "B"]
		labels = ["A", "B", "C"]
		binarizer = PMMLLabelBinarizer()
		self.assertFalse(hasattr(binarizer, "classes_"))
		binarizer.fit(y)
		self.assertEqual(labels, binarizer.classes_.tolist())
		binarizer.fit(numpy.array(y))
		self.assertEqual(labels, binarizer.classes_.tolist())
		binarizer.fit(Series(numpy.array(y)))
		self.assertEqual(labels, binarizer.classes_.tolist())

	def test_transform_float(self):
		y = [1.0, float("NaN"), 2.0, 3.0]
		binarizer = PMMLLabelBinarizer()
		binarizer.fit(y)
		self.assertEqual([[1, 0, 0], [0, 0, 1], [0, 0, 0], [0, 1, 0]], binarizer.transform([1.0, 3.0, float("NaN"), 2.0]).tolist())

	def test_transform_string(self):
		y = ["A", None, "B", "C"]
		binarizer = PMMLLabelBinarizer()
		binarizer.fit(y)
		self.assertEqual([[1, 0, 0], [0, 0, 1], [0, 0, 0], [0, 1, 0]], binarizer.transform(["A", "C", None, "B"]).tolist())
		self.assertEqual([[0, 0, 0]], binarizer.transform([None]).tolist())
		self.assertEqual([[1, 0, 0], [0, 1, 0], [0, 0, 1]], binarizer.transform(["A", "B", "C"]).tolist())

class PMMLLabelEncoderTest(TestCase):

	def test_fit_float(self):
		y = [1.0, float("NaN"), 1.0, 2.0, float("NaN"), 3.0, 3.0, 2.0]
		labels = [1.0, 2.0, 3.0]
		encoder = PMMLLabelEncoder(missing_value = -999)
		self.assertEqual(-999, encoder.missing_value)
		encoder.fit(y)
		self.assertEqual(labels, encoder.classes_.tolist())

	def test_fit_string(self):
		y = ["A", None, "A", "B", None, "C", "C", "B"]
		labels = ["A", "B", "C"]
		encoder = PMMLLabelEncoder()
		self.assertFalse(hasattr(encoder, "classes_"))
		encoder.fit(y)
		self.assertEqual(labels, encoder.classes_.tolist())
		encoder.fit(numpy.array(y))
		self.assertEqual(labels, encoder.classes_.tolist())
		encoder.fit(Series(numpy.array(y)))
		self.assertEqual(labels, encoder.classes_.tolist())

	def test_transform_float(self):
		y = [1.0, float("NaN"), 2.0, 3.0]
		encoder = PMMLLabelEncoder(missing_value = -999)
		encoder.fit(y)
		self.assertEqual([0, 2, -999, 1], encoder.transform([1.0, 3.0, float("NaN"), 2.0]).tolist())

	def test_transform_string(self):
		y = ["A", None, "B", "C"]
		encoder = PMMLLabelEncoder()
		encoder.fit(y)
		self.assertEqual([0, 2, None, 1], encoder.transform(["A", "C", None, "B"]).tolist())
		self.assertEqual([None], encoder.transform(numpy.array([None])).tolist())
		self.assertEqual([0, 1, 2], encoder.transform(Series(numpy.array(["A", "B", "C"]))).tolist())
