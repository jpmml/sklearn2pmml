from pandas import DataFrame, Series
from sklearn.dummy import DummyRegressor
from sklearn.feature_selection import f_regression, SelectFromModel, SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn2pmml import _classpath, _filter, _filter_steps, _strip_module, _supported_classes, make_pmml_pipeline, make_tpot_pmml_config, EstimatorProxy, SelectorProxy
from sklearn2pmml.pipeline import PMMLPipeline
from unittest import TestCase

import numpy

class EstimatorProxyTest(TestCase):

	def test_init(self):
		regressor = DummyRegressor()
		regressor.fit(numpy.array([[0], [0]]), numpy.array([0.0, 2.0]))
		self.assertEqual(1.0, regressor.constant_)
		regressor_proxy = EstimatorProxy(regressor, attr_names_ = ["constant_"])
		self.assertEqual(1.0, regressor_proxy.constant_)

	def test_fit(self):
		regressor = DummyRegressor()
		regressor_proxy = EstimatorProxy(regressor, attr_names_ = ["constant_"])
		self.assertFalse(hasattr(regressor_proxy, "constant_"))
		regressor_proxy.fit(numpy.array([[0], [0]]), numpy.array([0.0, 2.0]))
		self.assertEqual(1.0, regressor.constant_)
		self.assertEqual(1.0, regressor_proxy.constant_)

class SelectorProxyTest(TestCase):

	def test_init(self):
		selector = SelectKBest(score_func = f_regression, k = 1)
		selector.fit(numpy.array([[0, 0], [1.0, 2.0]]), numpy.array([0.5, 1.0]))
		self.assertEqual([0, 1], selector._get_support_mask().tolist())
		selector_proxy = SelectorProxy(selector)
		self.assertEqual([0, 1], selector_proxy.support_mask_.tolist())

	def test_fit(self):
		selector = SelectKBest(score_func = f_regression, k = 1)
		selector_proxy = SelectorProxy(selector)
		self.assertFalse(hasattr(selector_proxy, "support_mask_"))
		selector_proxy.fit(numpy.array([[0, 0], [1.0, 2.0]]), numpy.array([0.5, 1.0]))
		self.assertEqual([0, 1], selector._get_support_mask().tolist())
		self.assertEqual([0, 1], selector_proxy.support_mask_.tolist())

	def test_filter(self):
		selector = SelectFromModel(DecisionTreeRegressor(), prefit = False)
		self.assertIsInstance(selector, SelectFromModel)
		self.assertIsInstance(selector.estimator, DecisionTreeRegressor)
		self.assertFalse(hasattr(selector, "estimator_"))
		selector = _filter_steps([("selector", selector)])[0][1]
		self.assertIsInstance(selector, SelectorProxy)
		selector.fit(numpy.array([[0, 1], [0, 2], [0, 3]]), numpy.array([0.5, 1.0, 1.5]))
		self.assertIsInstance(selector.estimator, EstimatorProxy)
		self.assertIsInstance(selector.estimator_, EstimatorProxy)
		self.assertEqual([0, 1], selector._get_support_mask().tolist())

	def test_filter_prefit(self):
		regressor = DecisionTreeRegressor()
		regressor.fit(numpy.array([[0, 1], [0, 2], [0, 3]]), numpy.array([0.5, 1.0, 1.5]))
		selector = SelectFromModel(regressor, prefit = True)
		self.assertTrue(hasattr(selector, "estimator"))
		self.assertFalse(hasattr(selector, "estimator_"))
		selector = _filter_steps([("selector", selector, {})])[0][1]
		self.assertIsInstance(selector, SelectorProxy)
		self.assertIsInstance(selector.estimator, EstimatorProxy)
		self.assertFalse(hasattr(selector, "estimator_"))
		self.assertEqual([0, 1], selector._get_support_mask().tolist())

class ClasspathTest(TestCase):

	def test_classpath(self):
		classpath = _classpath([])
		self.assertEqual(20, len(classpath))
		classpath = _classpath(["A.jar", "B.jar"])
		self.assertEqual(20 + 2, len(classpath))

	def test_supported_classes(self):
		classes = _supported_classes([])
		self.assertTrue(len(classes) > 100)

	def test_strip_module(self):
		self.assertEqual("sklearn.decomposition.PCA", _strip_module("sklearn.decomposition.pca.PCA"))
		self.assertEqual("sklearn.feature_selection.SelectPercentile", _strip_module("sklearn.feature_selection.univariate_selection.SelectPercentile"))
		self.assertEqual("sklearn.preprocessing.StandardScaler", _strip_module("sklearn.preprocessing.data.StandardScaler"))
		self.assertEqual("sklearn.tree.DecisionTreeClassifier", _strip_module("sklearn.tree.tree.DecisionTreeClassifier"))

class FunctionTest(TestCase):

	def test_make_pmml_pipeline(self):
		estimator = DummyRegressor()
		pmml_pipeline = make_pmml_pipeline(estimator)
		self.assertTrue(isinstance(pmml_pipeline, PMMLPipeline))
		pipeline = Pipeline([
			("estimator", estimator)
		])
		pmml_pipeline = make_pmml_pipeline(pipeline)
		self.assertTrue(isinstance(pmml_pipeline, PMMLPipeline))

	def test_make_tpot_pmml_config(self):
		config = {
			"sklearn.kernel_approximation.RBFSampler" : {"gamma" : numpy.arange(0.0, 1.01, 0.05)},
			"sklearn.preprocessing.StandardScaler" : {}
		}
		tpot_pmml_config = make_tpot_pmml_config(config)
		self.assertEqual({"sklearn.preprocessing.StandardScaler" : {}}, tpot_pmml_config)
