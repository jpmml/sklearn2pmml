from pandas import DataFrame, Series
from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn2pmml.pipeline import _get_column_names, PMMLPipeline
from unittest import TestCase

import numpy

class PMMLPipelineTest(TestCase):

	def test_get_columns(self):
		X = DataFrame([[1, 0], [2, 0], [3, 0]], columns = [1, 2])
		self.assertEqual(["1", "2"], _get_column_names(X).tolist())
		X.columns = numpy.asarray([1.0, 2.0])
		self.assertEqual(["1.0", "2.0"], _get_column_names(X).tolist())
		x = Series([1, 2, 3], name = 1)
		self.assertEqual("1", _get_column_names(x).tolist())
		x.name = 1.0
		self.assertEqual("1.0", _get_column_names(x).tolist())

	def test_configure(self):
		regressor = DecisionTreeRegressor()
		pipeline = PMMLPipeline([("regressor", regressor)])
		self.assertFalse(hasattr(regressor, "pmml_options_"))
		pipeline.configure()
		self.assertFalse(hasattr(regressor, "pmml_options_"))
		pipeline.configure(compact = True, flat = True)
		self.assertTrue(hasattr(regressor, "pmml_options_"))
		pmml_options = regressor.pmml_options_
		self.assertEqual(True, pmml_options["compact"])
		self.assertEqual(True, pmml_options["flat"])

	def test_fit_verify(self):
		pipeline = PMMLPipeline([("estimator", DummyRegressor())])
		self.assertFalse(hasattr(pipeline, "active_fields"))
		self.assertFalse(hasattr(pipeline, "target_fields"))
		X = DataFrame([[1, 0], [2, 0], [3, 0]], columns = ["X1", "X2"])
		y = Series([0.5, 1.0, 1.5], name = "y")
		pipeline.fit(X, y)
		self.assertEqual(["X1", "X2"], pipeline.active_fields.tolist())
		self.assertEqual("y", pipeline.target_fields.tolist())
		X.columns = ["x1", "x2"]
		pipeline.fit(X, y)
		self.assertEqual(["x1", "x2"], pipeline.active_fields.tolist())
		self.assertEqual("y", pipeline.target_fields.tolist())
		self.assertFalse(hasattr(pipeline, "verification"))
		pipeline.verify(X.sample(2))
		self.assertEqual(2, len(pipeline.verification.active_values))
		self.assertEqual(2, len(pipeline.verification.target_values))
		X.columns = ["x2", "x1"]
		with self.assertRaises(ValueError):
			pipeline.verify(X.sample(2)) 
