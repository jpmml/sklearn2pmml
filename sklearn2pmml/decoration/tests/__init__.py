from pandas import DataFrame
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn2pmml.decoration import CategoricalDomain, ContinuousDomain
from sklearn_pandas import DataFrameMapper
from unittest import TestCase

import numpy

class CategoricalDomainTest(TestCase):

	def test_fit_int(self):
		domain = CategoricalDomain(missing_value_treatment = "as_value", missing_value_replacement = -999, invalid_value_treatment = "as_is")
		self.assertEqual("as_value", domain.missing_value_treatment)
		self.assertEqual(-999, domain.missing_value_replacement)
		self.assertEqual("as_is", domain.invalid_value_treatment)
		self.assertFalse(hasattr(domain, "data_"))
		self.assertFalse(hasattr(domain, "data_total_freq_"))
		self.assertFalse(hasattr(domain, "data_missing_freq_"))
		X = DataFrame(numpy.array([1, None, 3, 2, None, 2]))
		Xt = domain.fit_transform(X)
		self.assertEqual(numpy.array([1, 2, 3]).tolist(), domain.data_.tolist())
		self.assertEqual(numpy.array([1, -999, 3, 2, -999, 2]).tolist(), Xt[0].tolist())
		self.assertEqual(6, domain.data_total_freq_)
		self.assertEqual(2, domain.data_missing_freq_)
		X = numpy.array([None, None]);
		Xt = domain.transform(X)
		self.assertEqual(numpy.array([-999, -999]).tolist(), Xt.tolist())

	def test_fit_string(self):
		domain = CategoricalDomain()
		self.assertEqual("as_is", domain.missing_value_treatment)
		self.assertFalse(hasattr(domain, "missing_value_replacement"))
		self.assertEqual("return_invalid", domain.invalid_value_treatment)
		X = numpy.array(["1", None, "3", "2", None, "2"])
		Xt = domain.fit_transform(X)
		self.assertEqual(numpy.array(["1", "2", "3"]).tolist(), domain.data_.tolist())
		self.assertEqual(numpy.array(["1", None, "3", "2", None, "2"]).tolist(), Xt.tolist())

	def test_mapper(self):
		domain = CategoricalDomain()
		df = DataFrame([{"X" : "2", "y" : 2}, {"X" : "1"}, {"X" : "3"}])
		mapper = DataFrameMapper([
			("X", [domain, LabelBinarizer()]),
			("y", None)
		])
		mapper.fit_transform(df)
		self.assertEqual(numpy.array(["1", "2", "3"]).tolist(), domain.data_.tolist())
		self.assertEqual(3, domain.data_total_freq_)
		self.assertEqual(0, domain.data_missing_freq_)

class ContinuousDomainTest(TestCase):

	def test_fit_float(self):
		domain = ContinuousDomain(missing_value_treatment = "as_value", missing_value_replacement = -1.0, invalid_value_treatment = "as_is")
		self.assertEqual("as_value", domain.missing_value_treatment)
		self.assertEqual(-1.0, domain.missing_value_replacement)
		self.assertEqual("as_is", domain.invalid_value_treatment)
		self.assertFalse(hasattr(domain, "data_min_"))
		self.assertFalse(hasattr(domain, "data_max_"))
		self.assertFalse(hasattr(domain, "data_mean_"))
		self.assertFalse(hasattr(domain, "data_std_"))
		self.assertFalse(hasattr(domain, "data_median_"))
		self.assertFalse(hasattr(domain, "data_inter_quartile_range_"))
		self.assertFalse(hasattr(domain, "data_total_freq_"))
		self.assertFalse(hasattr(domain, "data_missing_freq_"))
		X = DataFrame(numpy.array([1.0, float('NaN'), 3.0, 2.0, float('NaN'), 2.0]))
		Xt = domain.fit_transform(X)
		self.assertEqual(1.0, domain.data_min_)
		self.assertEqual(3.0, domain.data_max_)
		self.assertEqual(2.0, domain.data_mean_)
		self.assertTrue(numpy.isclose(0.70710678118655, domain.data_std_))
		self.assertEqual(2.0, domain.data_median_)
		self.assertEqual(0.5, domain.data_inter_quartile_range_)
		self.assertEqual(6, domain.data_total_freq_)
		self.assertEqual(2, domain.data_missing_freq_)
		self.assertEqual(numpy.array([1.0, -1.0, 3.0, 2.0, -1.0, 2.0]).tolist(), Xt[0].tolist())
		X = numpy.array([float('NaN'), None])
		Xt = domain.transform(X)
		self.assertEqual(numpy.array([-1.0, -1.0]).tolist(), Xt.tolist())

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
		self.assertEqual(numpy.array([2.0, 2.0]).tolist(), domain.data_mean_.tolist())
		self.assertEqual(numpy.isclose(numpy.array([0.81649658092773, 1.2247448713916]), domain.data_std_).tolist(), [True, True])
		self.assertEqual(numpy.array([2.0, 2.0]).tolist(), domain.data_median_.tolist())
		self.assertEqual(numpy.array([1.0, 1.5]).tolist(), domain.data_inter_quartile_range_.tolist())
		self.assertEqual(numpy.array([3, 3]).tolist(), domain.data_total_freq_.tolist())
		self.assertEqual(numpy.array([0, 0]).tolist(), domain.data_missing_freq_.tolist())
