from pandas import DataFrame
from sklearn.preprocessing import Imputer, LabelBinarizer, StandardScaler
from sklearn2pmml.decoration import CategoricalDomain, ContinuousDomain
from sklearn_pandas import DataFrameMapper
from unittest import TestCase

import numpy

def _value_count(stats):
	return dict(zip(stats[0].tolist(), stats[1].tolist()))

def _array_to_list(info):
	return dict((k, v.tolist()) for k, v in info.items())

class CategoricalDomainTest(TestCase):

	def test_fit_int(self):
		domain = CategoricalDomain(missing_value_treatment = "as_value", missing_value_replacement = -999, invalid_value_treatment = "as_is")
		self.assertEqual("as_value", domain.missing_value_treatment)
		self.assertEqual(-999, domain.missing_value_replacement)
		self.assertEqual("as_is", domain.invalid_value_treatment)
		self.assertFalse(hasattr(domain, "data_"))
		self.assertFalse(hasattr(domain, "counts_"))
		self.assertFalse(hasattr(domain, "discr_stats_"))
		X = DataFrame(numpy.array([1, None, 3, 2, None, 2]))
		Xt = domain.fit_transform(X)
		self.assertEqual(numpy.array([1, 2, 3]).tolist(), domain.data_.tolist())
		self.assertEqual({"totalFreq" : 6, "missingFreq" : 2, "invalidFreq" : 0}, domain.counts_)
		self.assertEqual({1 : 1, 2 : 2, 3 : 1}, _value_count(domain.discr_stats_))
		self.assertEqual(numpy.array([1, -999, 3, 2, -999, 2]).tolist(), Xt[0].tolist())
		X = numpy.array([None, None]);
		Xt = domain.transform(X)
		self.assertEqual(numpy.array([-999, -999]).tolist(), Xt.tolist())

	def test_fit_string(self):
		domain = CategoricalDomain(with_statistics = False)
		self.assertEqual("as_is", domain.missing_value_treatment)
		self.assertFalse(hasattr(domain, "missing_value_replacement"))
		self.assertEqual("return_invalid", domain.invalid_value_treatment)
		X = numpy.array(["1", None, "3", "2", None, "2"])
		Xt = domain.fit_transform(X)
		self.assertEqual(numpy.array(["1", "2", "3"]).tolist(), domain.data_.tolist())
		self.assertFalse(hasattr(domain, "counts_"))
		self.assertFalse(hasattr(domain, "discr_stats_"))
		self.assertEqual(numpy.array(["1", None, "3", "2", None, "2"]).tolist(), Xt.tolist())

	def test_mapper(self):
		domain = CategoricalDomain()
		df = DataFrame([{"X" : "2", "y" : 2}, {"X" : "1"}, {"X" : "3"}])
		mapper = DataFrameMapper([
			("X", [domain, LabelBinarizer()]),
			("y", None)
		])
		mapper.fit_transform(df)
		self.assertEqual({"totalFreq" : 3, "missingFreq" : 0, "invalidFreq" : 0}, domain.counts_)
		self.assertEqual({"1" : 1, "2" : 1, "3" : 1}, _value_count(domain.discr_stats_))
		self.assertEqual(numpy.array(["1", "2", "3"]).tolist(), domain.data_.tolist())

class ContinuousDomainTest(TestCase):

	def test_fit_float(self):
		domain = ContinuousDomain(missing_value_treatment = "as_value", missing_value_replacement = -1.0, invalid_value_treatment = "as_is")
		self.assertEqual("as_value", domain.missing_value_treatment)
		self.assertEqual(-1.0, domain.missing_value_replacement)
		self.assertEqual("as_is", domain.invalid_value_treatment)
		self.assertFalse(hasattr(domain, "data_min_"))
		self.assertFalse(hasattr(domain, "data_max_"))
		self.assertFalse(hasattr(domain, "counts_"))
		self.assertFalse(hasattr(domain, "numeric_info_"))
		X = DataFrame(numpy.array([1.0, float('NaN'), 3.0, 2.0, float('NaN'), 2.0]))
		Xt = domain.fit_transform(X)
		self.assertEqual(1.0, domain.data_min_)
		self.assertEqual(3.0, domain.data_max_)
		self.assertEqual({"totalFreq" : 6, "missingFreq" : 2, "invalidFreq" : 0}, domain.counts_)
		self.assertEqual({"minimum" : [1.0], "maximum" : [3.0], "mean" : [2.0], "standardDeviation" : [0.7071067811865476], "median" : [2.0], "interQuartileRange" : [0.5]}, _array_to_list(domain.numeric_info_))
		self.assertEqual(numpy.array([1.0, -1.0, 3.0, 2.0, -1.0, 2.0]).tolist(), Xt[0].tolist())
		X = numpy.array([float('NaN'), None])
		Xt = domain.transform(X)
		self.assertEqual(numpy.array([-1.0, -1.0]).tolist(), Xt.tolist())

	def test_mapper(self):
		domain = ContinuousDomain()
		df = DataFrame([{"X1" : 2.0, "X2" : 2, "y" : 2.0}, {"X1" : 1.0, "X2" : 0.5}, {"X1" : 2}, {"X2" : 2}, {"X1" : 2.0, "y" : 1}, {"X1" : 3.0, "X2" : 3.5}])
		mapper = DataFrameMapper([
			(["X1", "X2"], [domain, Imputer(), StandardScaler()]),
			("y", None)
		])
		mapper.fit_transform(df)
		self.assertEqual({"totalFreq" : [6, 6], "missingFreq" : [1, 2], "invalidFreq" : [0, 0]}, _array_to_list(domain.counts_))
		self.assertEqual({"minimum" : [1.0, 0.5], "maximum" : [3.0, 3.5], "mean" : [2.0, 2.0]}, _array_to_list(dict((k, domain.numeric_info_[k]) for k in ["minimum", "maximum", "mean"])))
		self.assertEqual(numpy.array([1.0, 0.5]).tolist(), domain.data_min_.tolist())
		self.assertEqual(numpy.array([3.0, 3.5]).tolist(), domain.data_max_.tolist())
