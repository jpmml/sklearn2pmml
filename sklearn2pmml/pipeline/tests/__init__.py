from pandas import DataFrame, Series
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn2pmml.pipeline import _get_column_names, PMMLPipeline
from sklearn2pmml.util.pmml import Extension
from unittest import TestCase

import numpy

class PMMLPipelineTest(TestCase):

	def test_get_columns(self):
		X = DataFrame([[1, 0], [2, 0], [3, 0]], columns = [1, 2])
		self.assertEqual(["1", "2"], _get_column_names(X).tolist())
		X.columns = numpy.asarray([1.0, 2.0])
		self.assertEqual(["1.0", "2.0"], _get_column_names(X).tolist())
		X = Series([1, 2, 3], name = 1)
		self.assertEqual("1", _get_column_names(X).tolist())
		X.name = 1.0
		self.assertEqual("1.0", _get_column_names(X).tolist())

	def test_predict_transform(self):
		predict_transformer = FeatureUnion([
			("identity", FunctionTransformer(None)),
			("log10", FunctionTransformer(numpy.log10))
		])
		pipeline = PMMLPipeline([("estimator", DummyRegressor())], predict_transformer = predict_transformer)
		X = DataFrame([[1, 0], [2, 0], [3, 0]], columns = ["X1", "X2"])
		y = Series([0.5, 1.0, 1.5], name = "y")
		pipeline.fit(X, y)
		y_pred = [1.0, 1.0, 1.0]
		y_predt = [1.0, 1.0, numpy.log10(1.0)]
		self.assertEqual(y_pred, pipeline.predict(X).tolist())
		self.assertEqual([y_predt for i in range(0, 3)], pipeline.predict_transform(X).tolist())

	def test_predict_proba_transform(self):
		predict_proba_transformer = FunctionTransformer(numpy.log)
		pipeline = PMMLPipeline([("estimator", DummyClassifier(strategy = "prior"))], predict_proba_transformer = predict_proba_transformer)
		X = DataFrame([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], columns = ["x"])
		y = Series(["green", "red", "yellow", "green", "red", "green"], name = "y")
		pipeline.fit(X, y)
		self.assertEqual(["green", "red", "yellow"], pipeline._final_estimator.classes_.tolist())
		y_proba = [3 / 6.0, 2 / 6.0, 1 / 6.0]
		y_probat = [numpy.log(x) for x in y_proba]
		self.assertEqual([y_proba for i in range(0, 6)], pipeline.predict_proba(X).tolist())
		self.assertEqual([y_proba + y_probat for i in range(0, 6)], pipeline.predict_proba_transform(X).tolist())

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

	def test_configure(self):
		regressor = DecisionTreeRegressor()
		pipeline = PMMLPipeline([
			("pipeline", Pipeline([
				("regressor", regressor)
			]))
		])
		self.assertIsInstance(pipeline._final_estimator, Pipeline)
		self.assertIsInstance(pipeline._final_estimator._final_estimator, DecisionTreeRegressor)
		self.assertIs(pipeline._deep_final_estimator(), regressor)
		self.assertFalse(hasattr(regressor, "pmml_options_"))
		pipeline.configure()
		self.assertFalse(hasattr(regressor, "pmml_options_"))
		pipeline.configure(compact = True, flat = True)
		self.assertTrue(hasattr(regressor, "pmml_options_"))
		pmml_options = regressor.pmml_options_
		self.assertEqual(2, len(pmml_options))
		self.assertEqual(True, pmml_options["compact"])
		self.assertEqual(True, pmml_options["flat"])
		pipeline.configure(compact = False)
		pmml_options = regressor.pmml_options_
		self.assertEqual(2, len(pmml_options))
		self.assertEqual(False, pmml_options["compact"])
		self.assertEqual(True, pmml_options["flat"])
		pipeline.configure()
		self.assertFalse(hasattr(regressor, "pmml_options_"))

	def test_customize(self):
		regressor = DecisionTreeRegressor()
		pipeline = PMMLPipeline([
			("regressor", regressor)
		])
		extension = Extension(name = "name", value = "value")
		self.assertFalse(hasattr(regressor, "pmml_customizations_"))
		pipeline.customize(command = "insert", xpath_expr = None, pmml_element = extension)
		self.assertTrue(hasattr(regressor, "pmml_customizations_"))
		self.assertEqual(1, len(regressor.pmml_customizations_))
		extension = Extension(name = "name", value = "new value")
		pipeline.customize(command = "update", xpath_expr = "/:TreeModel/:Extension", pmml_element = extension)
		self.assertEqual(2, len(regressor.pmml_customizations_))
		pipeline.customize(command = "delete", xpath_expr = "/:TreeModel/:Extension")
		self.assertEqual(3, len(regressor.pmml_customizations_))
