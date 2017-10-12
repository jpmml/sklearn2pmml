from pandas import DataFrame, Series
from sklearn.dummy import DummyRegressor
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn2pmml import EstimatorProxy, PMMLPipeline, SelectorProxy
from unittest import TestCase

import numpy

class PMMLPipelineTest(TestCase):

	def test_fit(self):
		pipeline = PMMLPipeline([("estimator", DummyRegressor())])
		self.assertFalse(hasattr(pipeline, "active_fields"))
		self.assertFalse(hasattr(pipeline, "target_field"))
		X = DataFrame([[1, 0], [2, 0], [3, 0]], columns = ["X1", "X2"])
		y = Series([0.5, 1.0, 1.5], name = "y")
		pipeline.fit(X, y)
		self.assertEqual(["X1", "X2"], pipeline.active_fields.tolist())
		self.assertEqual("y", pipeline.target_field)
		X.columns = ["x1", "x2"]
		pipeline.fit(X, y)
		self.assertEqual(["x1", "x2"], pipeline.active_fields.tolist())

class EstimatorProxyTest(TestCase):

	def test_fit(self):
		regressor = DummyRegressor()
		regressor_proxy = EstimatorProxy(regressor, attr_names_ = ["constant_"])
		self.assertFalse(hasattr(regressor_proxy, "constant_"))
		regressor_proxy.fit(numpy.array([[0], [0]]), numpy.array([0.0, 2.0]))
		self.assertEqual(1.0, regressor.constant_)
		self.assertEqual(1.0, regressor_proxy.constant_)

class SelectorProxyTest(TestCase):

	def test_fit(self):
		selector = SelectKBest(score_func = f_regression, k = 1)
		selector_proxy = SelectorProxy(selector)
		self.assertFalse(hasattr(selector_proxy, "support_mask_"))
		selector_proxy.fit(numpy.array([[0, 0], [1.0, 2.0]]), numpy.array([0.5, 1.0]))
		self.assertEqual([0, 1], selector._get_support_mask().tolist())
		self.assertEqual([0, 1], selector_proxy.support_mask_.tolist())
