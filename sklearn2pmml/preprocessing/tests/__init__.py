from pandas import Series
from sklearn2pmml.preprocessing import PMMLLabelEncoder
from unittest import TestCase

import numpy

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
		self.assertEqual(numpy.array([0, 2, -999, 1]).tolist(), encoder.transform([1.0, 3.0, float("NaN"), 2.0]).tolist())

	def test_transform_string(self):
		y = ["A", None, "B", "C"]
		encoder = PMMLLabelEncoder()
		encoder.fit(y)
		self.assertEqual(numpy.array([0, 2, None, 1]).tolist(), encoder.transform(["A", "C", None, "B"]).tolist())
		self.assertEqual(numpy.array([None]).tolist(), encoder.transform(numpy.array([None])).tolist())
		self.assertEqual(numpy.array([0, 1, 2]).tolist(), encoder.transform(Series(numpy.array(["A", "B", "C"]))).tolist())
