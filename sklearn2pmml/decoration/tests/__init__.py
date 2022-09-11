from datetime import datetime
from pandas import Categorical, CategoricalDtype, DataFrame, Series
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, StandardScaler
from sklearn2pmml.decoration import Alias, CategoricalDomain, ContinuousDomain, ContinuousDomainEraser, DateDomain, DateTimeDomain, DiscreteDomainEraser, Domain, MultiDomain
from sklearn2pmml.preprocessing import ExpressionTransformer
from sklearn_pandas import DataFrameMapper
from unittest import TestCase

import numpy

class AliasTest(TestCase):

	def test_fit_transform(self):
		alias = Alias(StandardScaler(), name = "standard_scaler(X)")
		self.assertTrue(hasattr(alias, "transformer"))
		self.assertFalse(hasattr(alias, "transformer_"))
		alias.fit_transform(numpy.array([[0.0], [1.0], [2.0]]))
		self.assertTrue(hasattr(alias, "transformer_"))
		alias = Alias(StandardScaler(), name = "standard_scaler(X)", prefit = True)
		self.assertTrue(hasattr(alias, "transformer"))
		self.assertTrue(hasattr(alias, "transformer_"))

	def test_get_feature_names_out(self):
		X = DataFrame([[0, 0], [0, 1], [1, 1]], columns = ["x1", "x2"])
		alias = Alias(ExpressionTransformer("X[0] == X[1]", dtype = int), name = "flag", prefit = True)
		self.assertEqual(["flag"], alias.get_feature_names_out(None))
		ohe = OneHotEncoder()
		pipeline = Pipeline([
			("alias", alias),
			("ohe", ohe)
		])
		Xt = pipeline.fit_transform(X)
		self.assertEqual((3, 2), Xt.shape)
		if hasattr(pipeline, "get_feature_names_out"):
			self.assertEqual(["flag_0", "flag_1"], pipeline.get_feature_names_out(None).tolist())

def _value_count(stats):
	return dict(zip(stats[0].tolist(), stats[1].tolist()))

def _array_to_list(info):
	return dict((k, v.tolist()) for k, v in info.items())

class DomainTest(TestCase):

	def test_init(self):
		with self.assertRaises(ValueError):
			Domain(missing_value_treatment = "return_invalid", missing_value_replacement = 0)
		with self.assertRaises(ValueError):
			Domain(invalid_value_replacement = 0)
		with self.assertRaises(ValueError):
			Domain(invalid_value_treatment = "return_invalid", invalid_value_replacement = 0)

class CategoricalDomainTest(TestCase):

	def test_fit_int(self):
		domain = clone(CategoricalDomain(with_data = False, with_statistics = False))
		self.assertTrue(domain._empty_fit())
		domain = clone(CategoricalDomain(missing_value_treatment = "as_value", missing_value_replacement = 1, invalid_value_treatment = "as_value", invalid_value_replacement = 0))
		self.assertIsNone(domain.missing_values)
		self.assertEqual("as_value", domain.missing_value_treatment)
		self.assertEqual(1, domain.missing_value_replacement)
		self.assertEqual("as_value", domain.invalid_value_treatment)
		self.assertEqual(0, domain.invalid_value_replacement)
		self.assertFalse(hasattr(domain, "data_"))
		self.assertFalse(hasattr(domain, "counts_"))
		self.assertFalse(hasattr(domain, "discr_stats_"))
		self.assertFalse(domain._empty_fit())
		X = DataFrame([1, None, 3, 2, None, 2])
		Xt = domain.fit_transform(X)
		self.assertIsInstance(Xt, DataFrame)
		self.assertEqual([1, 2, 3], domain.data_.tolist())
		self.assertEqual({"totalFreq" : 6, "missingFreq" : 2, "invalidFreq" : 0}, domain.counts_)
		self.assertEqual({1 : 1, 2 : 2, 3 : 1}, _value_count(domain.discr_stats_))
		self.assertEqual([1, 1, 3, 2, 1, 2], Xt[0].tolist())
		X = numpy.array([None, None])
		Xt = domain.transform(X)
		self.assertEqual([1, 1], Xt.tolist())

	def test_fit_int_missing(self):
		domain = clone(CategoricalDomain(missing_values = -1, missing_value_replacement = 0, invalid_value_treatment = "as_missing"))
		self.assertEqual(-1, domain.missing_values)
		self.assertEqual(0, domain.missing_value_replacement)
		self.assertEqual("as_missing", domain.invalid_value_treatment)
		self.assertFalse(domain._empty_fit())
		X = DataFrame([1, -1, 3, 2, -1, 2])
		Xt = domain.fit_transform(X)
		self.assertIsInstance(Xt, DataFrame)
		self.assertEqual([0, 1, 2, 3], domain.data_.tolist())
		self.assertEqual({"totalFreq" : 6, "missingFreq" : 2, "invalidFreq" : 0}, domain.counts_)
		self.assertEqual({1 : 1, 2 : 2, 3 : 1}, _value_count(domain.discr_stats_))
		self.assertEqual([1, 0, 3, 2, 0, 2], Xt[0].tolist())
		X = numpy.array([-1, 4, 2])
		Xt = domain.transform(X)
		self.assertEqual([0, 0, 2], Xt.tolist())

	def test_fit_int_categorical(self):
		domain = clone(CategoricalDomain(dtype = CategoricalDtype()))
		self.assertIsNone(domain.dtype.categories)
		self.assertFalse(hasattr(domain, "dtype_"))
		X = Series([-1, 0, 1, 0, -1])
		Xt = domain.fit_transform(X)
		self.assertIsNone(domain.dtype.categories)
		self.assertEqual([-1, 0, 1], domain.dtype_.categories.tolist())

	def test_fit_string(self):
		domain = clone(CategoricalDomain(with_data = False, with_statistics = False))
		self.assertTrue(domain._empty_fit())
		domain = clone(CategoricalDomain(missing_values = None, with_statistics = False))
		self.assertIsNone(domain.missing_values)
		self.assertEqual("as_is", domain.missing_value_treatment)
		self.assertIsNone(domain.missing_value_replacement)
		self.assertEqual("return_invalid", domain.invalid_value_treatment)
		self.assertIsNone(domain.invalid_value_replacement)
		self.assertFalse(domain._empty_fit())
		X = DataFrame(["1", None, "3", "2", None, "2"])
		Xt = domain.fit_transform(X)
		self.assertIsInstance(Xt, DataFrame)
		self.assertEqual(["1", "2", "3"], domain.data_.tolist())
		self.assertFalse(hasattr(domain, "counts_"))
		self.assertFalse(hasattr(domain, "discr_stats_"))
		self.assertEqual(["1", None, "3", "2", None, "2"], Xt.iloc[:, 0].tolist())
		X = numpy.array([None, None])
		Xt = domain.transform(X)
		self.assertEqual([None, None], Xt.tolist())
		X = numpy.array(["4"])
		with self.assertRaises(ValueError):
			domain.transform(X)

	def test_fit_string_missing(self):
		domain = clone(CategoricalDomain(missing_values = ["NA", "N/A"], missing_value_replacement = "0", invalid_value_treatment = "as_value", invalid_value_replacement = "1"))
		self.assertEqual(["NA", "N/A"], domain.missing_values)
		self.assertEqual("0", domain.missing_value_replacement)
		self.assertEqual("as_value", domain.invalid_value_treatment)
		self.assertEqual("1", domain.invalid_value_replacement)
		self.assertFalse(domain._empty_fit())
		X = DataFrame(["1", "NA", "3", "2", "N/A", "2"])
		Xt = domain.fit_transform(X)
		self.assertIsInstance(Xt, DataFrame)
		self.assertEqual(["0", "1", "2", "3"], domain.data_.tolist())
		self.assertEqual({"totalFreq" : 6, "missingFreq" : 2, "invalidFreq" : 0}, domain.counts_)
		self.assertEqual({"1" : 1, "2" : 2, "3" : 1}, _value_count(domain.discr_stats_))
		self.assertEqual(["1", "0", "3", "2", "0", "2"], Xt.iloc[:, 0].tolist())
		X = numpy.array(["NA", "N/A", "4"])
		Xt = domain.transform(X)
		self.assertEqual(["0", "0", "1"], Xt.tolist())

	def test_fit_string_categorical(self):
		domain = clone(CategoricalDomain())
		X = Categorical(["a", "b", "c", "b", "a"])
		Xt = domain.fit_transform(X)
		self.assertIsInstance(Xt, Categorical)
		self.assertEqual(["a", "b", "c"], domain.data_.tolist())
		X = Categorical(X.tolist(), dtype = CategoricalDtype(categories = ["c", "b", "a"]))
		Xt = domain.fit_transform(X)
		self.assertIsInstance(Xt, Categorical)
		self.assertEqual(["c", "b", "a"], domain.data_.tolist())
		domain = clone(CategoricalDomain(dtype = CategoricalDtype()))
		X = Series(["a", "b", "c"])
		Xt = domain.fit_transform(X)
		self.assertIsInstance(Xt, Series)
		self.assertIsInstance(domain.dtype, CategoricalDtype)
		self.assertIsInstance(domain.dtype_, CategoricalDtype)
		self.assertEqual(["a", "b", "c"], domain.data_.tolist())
		domain = clone(CategoricalDomain(dtype = CategoricalDtype(categories = ["c", "b", "a"])))
		Xt = domain.fit_transform(X)
		self.assertEqual(["c", "b", "a"], domain.data_.tolist())

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
		self.assertEqual(["1", "2", "3"], domain.data_.tolist())

	def test_mapper_categorical(self):
		domain = CategoricalDomain(dtype = CategoricalDtype())
		str_domain = clone(domain)
		int_domain = clone(domain)
		df = DataFrame([["a", -1], ["b", 0], ["c", 1], ["b", 0], ["a", -1]], columns = ["str", "int"])
		self.assertEqual(numpy.dtype("object"), df["str"].dtype)
		self.assertEqual(int, df["int"].dtype)
		mapper = DataFrameMapper([
			(["str"], str_domain),
			(["int"], int_domain)
		], input_df = True, df_out = True)
		dft = mapper.fit_transform(df)
		self.assertIsInstance(dft, DataFrame)
		self.assertEqual(["a", "b", "c"], str_domain.data_.tolist())
		self.assertIsInstance(str_domain.dtype_, CategoricalDtype)
		self.assertEqual([-1, 0, 1], int_domain.data_.tolist())
		self.assertIsInstance(int_domain.dtype_, CategoricalDtype)
		self.assertEqual(CategoricalDtype(categories = ["a", "b", "c"]), dft["str"].dtype)
		self.assertEqual(CategoricalDtype(categories = [-1, 0, 1]), dft["int"].dtype)

class ContinuousDomainTest(TestCase):

	def test_fit_float(self):
		domain = clone(ContinuousDomain(with_data = False, with_statistics = False))
		self.assertTrue(domain._empty_fit())
		domain = clone(ContinuousDomain(missing_values = float("NaN"), missing_value_treatment = "as_value", missing_value_replacement = -1.0, invalid_value_treatment = "as_value", invalid_value_replacement = 0.0))
		self.assertTrue(numpy.isnan(domain.missing_values))
		self.assertEqual("as_value", domain.missing_value_treatment)
		self.assertEqual(-1.0, domain.missing_value_replacement)
		self.assertEqual("as_value", domain.invalid_value_treatment)
		self.assertEqual(0.0, domain.invalid_value_replacement)
		self.assertFalse(hasattr(domain, "data_min_"))
		self.assertFalse(hasattr(domain, "data_max_"))
		self.assertFalse(hasattr(domain, "counts_"))
		self.assertFalse(hasattr(domain, "numeric_info_"))
		self.assertFalse(domain._empty_fit())
		X = DataFrame([1.0, float("NaN"), 3.0, 2.0, float("NaN"), 2.0])
		Xt = domain.fit_transform(X)
		self.assertIsInstance(Xt, DataFrame)
		self.assertEqual(1.0, domain.data_min_)
		self.assertEqual(3.0, domain.data_max_)
		self.assertEqual({"totalFreq" : 6, "missingFreq" : 2, "invalidFreq" : 0}, domain.counts_)
		self.assertEqual({"minimum" : [1.0], "maximum" : [3.0], "mean" : [2.0], "standardDeviation" : [0.7071067811865476], "median" : [2.0], "interQuartileRange" : [0.5]}, _array_to_list(domain.numeric_info_))
		self.assertEqual([1.0, -1.0, 3.0, 2.0, -1.0, 2.0], Xt[0].tolist())
		X = numpy.array([float("NaN"), 4.0])
		Xt = domain.transform(X)
		self.assertEqual([-1.0, 0.0], Xt.tolist())

	def test_fit_float_missing(self):
		domain = clone(ContinuousDomain(missing_values = [-999.0, -1.0], missing_value_treatment = "as_value", missing_value_replacement = 0.0, invalid_value_treatment = "as_missing"))
		self.assertEqual([-999.0, -1.0], domain.missing_values)
		self.assertEqual("as_value", domain.missing_value_treatment)
		self.assertEqual(0.0, domain.missing_value_replacement)
		self.assertEqual("as_missing", domain.invalid_value_treatment)
		self.assertFalse(domain._empty_fit())
		X = DataFrame([1.0, -999.0, 3.0, -1.0, 2.0, -1.0, 2.0])
		Xt = domain.fit_transform(X)
		self.assertIsInstance(Xt, DataFrame)
		self.assertEqual(1.0, domain.data_min_)
		self.assertEqual(3.0, domain.data_max_)
		self.assertEqual({"totalFreq" : 7, "missingFreq" : 3, "invalidFreq" : 0}, domain.counts_)
		self.assertEqual({"minimum" : [1.0], "maximum" : [3.0], "mean" : [2.0], "standardDeviation" : [0.7071067811865476], "median" : [2.0], "interQuartileRange" : [0.5]}, _array_to_list(domain.numeric_info_))
		self.assertEqual([1.0, 0.0, 3.0, 0.0, 2.0, 0.0, 2.0], Xt[0].tolist())
		X = numpy.array([-999.0, -1.0, 4.0])
		Xt = domain.transform(X)
		self.assertEqual([0.0, 0.0, 0.0], Xt.tolist())

	def test_fit_float_outlier(self):
		domain = clone(ContinuousDomain(missing_values = float("NaN"), missing_value_replacement = 1.0, outlier_treatment = "as_missing_values", low_value = 0.0, high_value = 3.0))
		self.assertEqual(0.0, domain.low_value)
		self.assertEqual(3.0, domain.high_value)
		X = DataFrame([[-2.0, float("NaN")], [2.0, 4.0], [float("NaN"), 0.0]])
		mask = domain._missing_value_mask(X)
		self.assertEqual([[False, True], [False, False], [True, False]], mask.values.tolist())
		self.assertEqual([[True, False], [False, True], [False, False]], domain._outlier_mask(X, ~mask).values.tolist())
		self.assertEqual([[True, False], [False, False], [False, False]], domain._negative_outlier_mask(X, ~mask).values.tolist())
		self.assertEqual([[False, False], [False, True], [False, False]], domain._positive_outlier_mask(X, ~mask).values.tolist())
		Xt = domain.fit_transform(X)
		self.assertEqual([1.0, 2.0, 1.0], Xt[0].tolist())
		self.assertEqual([1.0, 1.0, 0.0], Xt[1].tolist())
		domain = clone(ContinuousDomain(outlier_treatment = "as_extreme_values", low_value = 0.0, high_value = 3.0, missing_values = -1.0))
		X = DataFrame([[-2.0, -1.0], [2.0, 4.0], [-1.0, 0.0]])
		mask = domain._missing_value_mask(X)
		self.assertEqual([[False, True], [False, False], [True, False]], mask.values.tolist())
		self.assertEqual([[True, False], [False, True], [False, False]], domain._outlier_mask(X, ~mask).values.tolist())
		self.assertEqual([[True, False], [False, False], [False, False]], domain._negative_outlier_mask(X, ~mask).values.tolist())
		self.assertEqual([[False, False], [False, True], [False, False]], domain._positive_outlier_mask(X, ~mask).values.tolist())
		Xt = domain.fit_transform(X)
		self.assertEqual([0.0, 2.0, -1.0], X[0].tolist())
		self.assertEqual([-1.0, 3.0, 0.0], X[1].tolist())

	def test_mapper(self):
		domain = ContinuousDomain()
		df = DataFrame([{"X1" : 2.0, "X2" : 2, "y" : 2.0}, {"X1" : 1.0, "X2" : 0.5}, {"X1" : 2}, {"X2" : 2}, {"X1" : 2.0, "y" : 1}, {"X1" : 3.0, "X2" : 3.5}])
		mapper = DataFrameMapper([
			(["X1", "X2"], [domain, SimpleImputer(), StandardScaler()]),
			("y", None)
		])
		mapper.fit_transform(df)
		self.assertEqual({"totalFreq" : [6, 6], "missingFreq" : [1, 2], "invalidFreq" : [0, 0]}, _array_to_list(domain.counts_))
		self.assertEqual({"minimum" : [1.0, 0.5], "maximum" : [3.0, 3.5], "mean" : [2.0, 2.0]}, _array_to_list(dict((k, domain.numeric_info_[k]) for k in ["minimum", "maximum", "mean"])))
		self.assertEqual([1.0, 0.5], domain.data_min_.tolist())
		self.assertEqual([3.0, 3.5], domain.data_max_.tolist())

class TemporalDomainTest(TestCase):

	def test_fit_transform(self):
		X = DataFrame([["1959-12-31", "1959-12-31T23:59:59"], ["1960-01-01", "1960-01-01T01:01:10"], ["2003-04-01", "2003-04-01T05:16:27"]], columns = ["date", "datetime"])
		domain = clone(DateDomain())
		Xt = domain.fit_transform(X)
		self.assertEqual([datetime(1959, 12, 31), datetime(1960, 1, 1), datetime(2003, 4, 1)], Xt["date"].tolist())
		self.assertEqual([datetime(1959, 12, 31), datetime(1960, 1, 1), datetime(2003, 4, 1)], Xt["datetime"].tolist())
		Xt = domain.fit_transform(X.values)
		self.assertEqual([datetime(1959, 12, 31), datetime(1960, 1, 1), datetime(2003, 4, 1)], Xt[:, 0].tolist())
		domain = clone(DateTimeDomain())
		Xt = domain.fit_transform(X)
		self.assertEqual([datetime(1959, 12, 31, 0, 0, 0), datetime(1960, 1, 1, 0, 0, 0), datetime(2003, 4, 1, 0, 0, 0)], Xt["date"].tolist())
		self.assertEqual([datetime(1959, 12, 31, 23, 59, 59), datetime(1960, 1, 1, 1, 1, 10), datetime(2003, 4, 1, 5, 16, 27)], Xt["datetime"].tolist())
		Xt = domain.fit_transform(X.values)
		self.assertEqual([datetime(1959, 12, 31, 0, 0, 0), datetime(1960, 1, 1, 0, 0, 0), datetime(2003, 4, 1, 0, 0, 0)], Xt[:, 0].tolist())

class MultiDomainTest(TestCase):

	def test_fit_transform(self):
		domain = clone(MultiDomain([ContinuousDomain(missing_value_replacement = 0.0), None, CategoricalDomain(missing_value_replacement = "zero")]))
		X = DataFrame([[-1.0, -1, "minus one"], [float("NaN"), 0, None], [1.0, 1, "one"]], columns = ["x1", "x2", "x3"])
		Xt = domain.fit_transform(X)
		self.assertTrue(isinstance(Xt, DataFrame))
		self.assertEqual([-1.0, 0.0, 1.0], Xt["x1"].tolist())
		self.assertEqual([-1, 0, 1], Xt["x2"].tolist())
		self.assertEqual(["minus one", "zero", "one"], Xt["x3"].tolist())
		X = numpy.array([[float("NaN"), 0, None]])
		Xt = domain.transform(X)
		self.assertTrue(isinstance(Xt, numpy.ndarray))
		self.assertTrue([0.0], Xt[:, 0].tolist())
		self.assertTrue([0], Xt[:, 1].tolist())
		self.assertTrue(["zero"], Xt[:, 2].tolist())
