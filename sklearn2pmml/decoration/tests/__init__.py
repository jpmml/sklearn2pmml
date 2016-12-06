from pandas import DataFrame
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn2pmml.decoration import CategoricalDomain, ContinuousDomain
from sklearn_pandas import DataFrameMapper
from unittest import TestCase

import numpy

class CategoricalDomainTest(TestCase):

	def test_fit_int(self):
		domain = CategoricalDomain()
		self.assertEqual("return_invalid", domain.invalid_value_treatment)
		self.assertFalse(hasattr(domain, "data_"))
		domain = domain.fit(numpy.array([1, 3, 2, 2]))
		self.assertEqual(numpy.array([1, 2, 3]).tolist(), domain.data_.tolist())

	def test_fit_string(self):
		domain = CategoricalDomain()
		domain = domain.fit(numpy.array(["1", None, "3", "2", None, "2"]))
		self.assertEqual(numpy.array(["1", "2", "3"]).tolist(), domain.data_.tolist())

	def test_mapper(self):
		domain = CategoricalDomain()
		df = DataFrame([{"X" : "2", "y" : 2}, {"X" : "1"}, {"X" : "3"}])
		mapper = DataFrameMapper([
			("X", [domain, LabelBinarizer()]),
			("y", None)
		])
		mapper.fit_transform(df)
		self.assertEqual(numpy.array(["1", "2", "3"]).tolist(), domain.data_.tolist())

class ContinuousDomainTest(TestCase):

	def test_fit_float(self):
		domain = ContinuousDomain()
		self.assertEqual("return_invalid", domain.invalid_value_treatment)
		self.assertFalse(hasattr(domain, "data_min_"))
		self.assertFalse(hasattr(domain, "data_max_"))
		domain = domain.fit(numpy.array([1.0, float('NaN'), 3.0, 2.0, float('NaN'), 2.0]))
		self.assertEqual(1.0, domain.data_min_)
		self.assertEqual(3.0, domain.data_max_)

	def test_mapper(self):
		domain = ContinuousDomain()
		df = DataFrame([{"X1" : 2.0, "X2" : 2, "y" : 2.0}, {"X1" : 1.0, "X2" : 0.5}, {"X1" : 3.0, "X2" : 3.5}])
		mapper = DataFrameMapper([
			(["X1", "X2"], [domain, StandardScaler()]),
			("y", None)
		])
		mapper.fit_transform(df)
		self.assertEqual(numpy.array([1.0, 0.5]).tolist(), domain.data_min_.tolist())
		self.assertEqual(numpy.array([3.0, 3.5]).tolist(), domain.data_max_.tolist())
