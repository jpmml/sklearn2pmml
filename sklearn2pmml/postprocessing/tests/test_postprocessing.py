from pandas import DataFrame
from sklearn2pmml.postprocessing import FeatureExporter, BusinessDecisionTransformer
from sklearn2pmml.preprocessing import ExpressionTransformer
from unittest import TestCase

import numpy

class BusinessDecisionTransformerTest(TestCase):

	def test_transform(self):
		business_problem = "Should the action be taken?"
		decisions = [
			(False, "Do not take the action"),
			(True, "Take the action")
		]
		transformer = BusinessDecisionTransformer(transformer = None, business_problem = business_problem, decisions = decisions, prefit = False)
		self.assertTrue(hasattr(transformer, "transformer"))
		self.assertFalse(hasattr(transformer, "transformer_"))
		X = numpy.asarray([[False], [True]])
		self.assertEqual([[False], [True]], transformer.fit_transform(X).tolist())
		self.assertTrue(hasattr(transformer, "transformer_"))
		self.assertIsNone(transformer.transformer_)
		transformer = BusinessDecisionTransformer(transformer = "X[1]", business_problem = business_problem, decisions = decisions, prefit = True)
		self.assertTrue(hasattr(transformer, "transformer"))
		self.assertTrue(hasattr(transformer, "transformer_"))
		self.assertIsInstance(transformer.transformer_, ExpressionTransformer)
		X = numpy.asarray([[0, False], [1, True]])
		self.assertEqual([[False], [True]], transformer.transform(X).tolist())
		self.assertEqual([[False], [True]], transformer.fit_transform(X).tolist())
		transformer = BusinessDecisionTransformer(transformer = ExpressionTransformer("X[1]"), business_problem = business_problem, decisions = decisions, prefit = True)
		self.assertTrue(hasattr(transformer, "transformer"))
		self.assertTrue(hasattr(transformer, "transformer_"))
		self.assertEqual([[False], [True]], transformer.transform(X).tolist())
		self.assertEqual([[False], [True]], transformer.fit_transform(X).tolist())

class FeatureExporterTest(TestCase):

	def test_transform(self):
		X = DataFrame([["a", 1], ["b", 2], ["c", 3], ["d", 4]], columns = ["x1", "x2"])
		transformer = FeatureExporter(names = ["x(1)", "x(2)"])
		self.assertEqual(["x1", "x2"], transformer.get_feature_names_out(X.columns).tolist())
		Xt = transformer.transform(X)
		self.assertIsInstance(Xt, DataFrame)
		self.assertEqual((4, 2), Xt.shape)
		X = X.to_numpy()
		Xt = transformer.transform(X)
		self.assertIsInstance(Xt, numpy.ndarray)
		self.assertEqual((4, 2), Xt.shape)
