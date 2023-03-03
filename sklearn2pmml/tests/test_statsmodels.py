from sklearn.datasets import load_diabetes, load_iris
from sklearn2pmml.statsmodels import StatsModelsClassifier, StatsModelsRegressor
from statsmodels.api import Logit, MNLogit, OLS
from unittest import TestCase

import numpy

class StatsModelsClassifierTest(TestCase):

	def test_binary_classification(self):
		iris_X, iris_y = load_iris(return_X_y = True)
		iris_y = (iris_y == 1)
		classifier = StatsModelsClassifier(Logit)
		self.assertTrue(hasattr(classifier, "model_class"))
		self.assertTrue(hasattr(classifier, "fit_intercept"))
		classifier.fit(iris_X, iris_y)
		self.assertTrue(hasattr(classifier, "model_"))
		self.assertTrue(hasattr(classifier, "results_"))
		self.assertIsInstance(classifier.model_, Logit)
		self.assertEqual([False, True], classifier.classes_.tolist())
		classifier.remove_data()
		species = classifier.predict(iris_X)
		self.assertEqual((150, ), species.shape)
		self.assertEqual(classifier.classes_.tolist(), numpy.unique(species).tolist())
		species_proba = classifier.predict_proba(iris_X)
		self.assertEqual((150, 2), species_proba.shape)
		self.assertEqual(150, numpy.sum(species_proba))

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
		species = classifier.predict(iris_X)
		self.assertEqual((150, ), species.shape)
		self.assertEqual(classifier.classes_.tolist(), numpy.unique(species).tolist())
		species_proba = classifier.predict_proba(iris_X)
		self.assertEqual((150, 3), species_proba.shape)
		self.assertEqual(150, numpy.sum(species_proba))

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
