from pandas import CategoricalDtype, Series
from sklearn.datasets import load_diabetes, load_iris
from sklearn.preprocessing import KBinsDiscretizer
from sklearn2pmml.statsmodels import StatsModelsClassifier, StatsModelsOrdinalClassifier, StatsModelsRegressor
from statsmodels.api import Logit, MNLogit, OLS
from statsmodels.miscmodels.ordinal_model import OrderedModel
from unittest import TestCase

import numpy

class StatsModelsClassifierTest(TestCase):

	def test_binary_classification(self):
		iris_X, iris_y = load_iris(return_X_y = True)
		iris_y = (iris_y == 1)
		classifier = StatsModelsClassifier(Logit)
		self.assertTrue(hasattr(classifier, "model_class"))
		self.assertTrue(hasattr(classifier, "fit_intercept"))
		classifier.fit(iris_X, iris_y, fit_method = "fit_regularized")
		self.assertTrue(hasattr(classifier, "model_"))
		self.assertTrue(hasattr(classifier, "results_"))
		self.assertIsInstance(classifier.model_, Logit)
		self.assertEqual([False, True], classifier.classes_.tolist())
		classifier.remove_data()
		pred = classifier.predict(iris_X)
		self.assertEqual((150, ), pred.shape)
		self.assertEqual(classifier.classes_.tolist(), numpy.unique(pred).tolist())
		pred_proba = classifier.predict_proba(iris_X)
		self.assertEqual((150, 2), pred_proba.shape)
		self.assertEqual(150, numpy.sum(pred_proba))

	def test_binary_classification_shape(self):
		iris_X, iris_y = load_iris(return_X_y = True)
		iris_y = (iris_y == 1)
		classifier = StatsModelsClassifier(Logit, fit_intercept = False)
		classifier.fit(iris_X, iris_y)
		self.assertEqual((1, 4), classifier.coef_.shape)
		self.assertEqual((1, ), classifier.intercept_.shape)
		self.assertEqual([0], classifier.intercept_.tolist())
		classifier = StatsModelsClassifier(Logit, fit_intercept = True)
		classifier.fit(iris_X, iris_y)
		self.assertEqual((1, 4), classifier.coef_.shape)
		self.assertEqual((1, ), classifier.intercept_.shape)

	def test_multiclass_classification(self):
		iris_X, iris_y = load_iris(return_X_y = True)
		classifier = StatsModelsClassifier(MNLogit)
		self.assertTrue(hasattr(classifier, "model_class"))
		self.assertTrue(hasattr(classifier, "fit_intercept"))
		# See https://stackoverflow.com/a/31511894
		classifier.fit(iris_X, iris_y, method = "bfgs")
		self.assertTrue(hasattr(classifier, "model_"))
		self.assertTrue(hasattr(classifier, "results_"))
		self.assertIsInstance(classifier.model_, MNLogit)
		self.assertEqual([0, 1, 2], classifier.classes_.tolist())
		classifier.remove_data()
		pred = classifier.predict(iris_X)
		self.assertEqual((150, ), pred.shape)
		self.assertEqual(classifier.classes_.tolist(), numpy.unique(pred).tolist())
		pred_proba = classifier.predict_proba(iris_X)
		self.assertEqual((150, 3), pred_proba.shape)
		self.assertEqual(150, numpy.sum(pred_proba))

	def test_multiclass_classification_shape(self):
		iris_X, iris_y = load_iris(return_X_y = True)
		classifier = StatsModelsClassifier(MNLogit, fit_intercept = False)
		classifier.fit(iris_X, iris_y)
		self.assertEqual((2, 4), classifier.coef_.shape)
		self.assertEqual((2, ), classifier.intercept_.shape)
		self.assertEqual([0, 0], classifier.intercept_.tolist())
		classifier = StatsModelsClassifier(MNLogit, fit_intercept = True)
		classifier.fit(iris_X, iris_y)
		self.assertEqual((2, 4), classifier.coef_.shape)
		self.assertEqual((2, ), classifier.intercept_.shape)

class StatsModelsOrdinalClassifierTest(TestCase):

	def test_ordinal_classification(self):
		diabetes_X, diabetes_y = load_diabetes(return_X_y = True)
		discretizer = KBinsDiscretizer(n_bins = 5, encode = "ordinal", strategy = "kmeans")
		diabetes_y = discretizer.fit_transform(diabetes_y.reshape((-1, 1))).astype(int)
		diabetes_y = numpy.vectorize(lambda x:"c{}".format(x + 1))(diabetes_y)
		diabetes_y = Series(diabetes_y.ravel(), dtype = CategoricalDtype(["c1", "c2", "c3", "c4", "c5"], ordered = True))
		classifier = StatsModelsOrdinalClassifier(OrderedModel, distr = "logit")
		classifier.fit(diabetes_X, diabetes_y)
		self.assertTrue(hasattr(classifier, "model_"))
		self.assertTrue(hasattr(classifier, "results_"))
		self.assertEqual(["c1", "c2", "c3", "c4", "c5"], classifier.classes_.tolist())
		self.assertEqual((10, ), classifier.coef_.shape)
		self.assertEquals(0, classifier.intercept_)
		self.assertEqual((4, ), classifier.threshold_.shape)
		classifier.remove_data()
		pred = classifier.predict(diabetes_X)
		self.assertEqual((442, ), pred.shape)
		pred_proba = classifier.predict_proba(diabetes_X)
		self.assertEqual((442, 5), pred_proba.shape)

class StatsModelsRegressorTest(TestCase):

	def test_regression(self):
		diabetes_X, diabetes_y = load_diabetes(return_X_y = True)
		regressor = StatsModelsRegressor(OLS)
		self.assertTrue(hasattr(regressor, "model_class"))
		self.assertTrue(hasattr(regressor, "fit_intercept"))
		regressor.fit(diabetes_X, diabetes_y)
		self.assertTrue(hasattr(regressor, "model_"))
		self.assertTrue(hasattr(regressor, "results_"))
		self.assertIsInstance(regressor.model_, OLS)

	def test_regression_shape(self):
		diabetes_X, diabetes_y = load_diabetes(return_X_y = True)
		regressor = StatsModelsRegressor(OLS, fit_intercept = False)
		regressor.fit(diabetes_X, diabetes_y)
		self.assertEqual((10, ), regressor.coef_.shape)
		self.assertEqual(0, regressor.intercept_)
		regressor = StatsModelsRegressor(OLS, fit_intercept = True)
		regressor.fit(diabetes_X, diabetes_y)
		self.assertEqual((10, ), regressor.coef_.shape)
		self.assertNotEqual(0, regressor.intercept_)
