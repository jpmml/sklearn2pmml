from pandas import DataFrame
from sklearn.base import clone
from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegression, SGDClassifier, SGDRegressor
from sklearn.svm import LinearSVC
from sklearn2pmml.ensemble import _checkLM, _checkLR, _step_params, MultiEstimatorChain, SelectFirstClassifier, SelectFirstRegressor

from unittest import TestCase

class GBDTLRTest(TestCase):

	def test_lm(self):
		_checkLM(ElasticNet())
		_checkLM(LinearRegression())
		_checkLM(SGDRegressor())

	def test_lr(self):
		_checkLR(LinearSVC())
		_checkLR(LogisticRegression())
		_checkLR(SGDClassifier())

	def test_step_params(self):
		params = {
			"gbdt__first" : 1,
			"lr__first" : 1.0,
			"gbdt__second" : 2,
			"any__any" : None
		}
		gbdt_params = _step_params("gbdt", params)
		self.assertEqual({"first" : 1, "second" : 2}, gbdt_params)
		self.assertEqual({"lr__first" : 1.0, "any__any" : None}, params)
		lr_params = _step_params("lr", params)
		self.assertEqual({"first" : 1.0}, lr_params)
		self.assertEqual({"any__any" : None}, params)

class MultiEstimatorChainTest(TestCase):

	def test_fit_predict(self):
		df = DataFrame([[-1, 0], [0, 0], [-1, -1], [1, 1], [-1, -1]], columns = ["X", "y"])
		X = df[["X"]]
		y = df["y"]
		estimator = MultiEstimatorChain([
			("negative", DummyClassifier(strategy = "most_frequent"), "X[0] < 0"),
			("not_negative", DummyClassifier(strategy = "most_frequent"), "X[0] >= 0"),
			("any", DummyClassifier(strategy = "most_frequent"), str(True))
		])
		params = estimator.get_params(deep = True)
		self.assertEqual("most_frequent", params["negative__strategy"])
		self.assertEqual("most_frequent", params["not_negative__strategy"])
		self.assertEqual("most_frequent", params["any__strategy"])
		estimator.fit(X, y)
		preds = estimator.predict(X)
		self.assertEqual((5, 3), preds.shape)
		self.assertEqual([-1, None, -1, None, -1], preds[:, 0].tolist())
		self.assertEqual([None, 0, None, 0, None], preds[:, 1].tolist())
		self.assertEqual([-1, -1, -1, -1, -1], preds[:, 2].tolist())

	def test_complex_fit_predict(self):
		X, y = load_iris(return_X_y = True)
		classifier = MultiEstimatorChain.Link(LogisticRegression(), augment_funcs = ["predict", "predict_proba"])
		regressor = SelectFirstRegressor([
			("not_setosa", LinearRegression(), "X[-3] < 0.5"),
			("setosa", LinearRegression(), "X[-3] >= 0.5")
		])
		estimator = MultiEstimatorChain([
			("classifier", classifier, str(True)),
			("regressor", regressor, str(True))
		])
		estimator.fit(X, y)
		self.assertEqual((3, 4), classifier.estimator_.coef_.shape)
		self.assertEqual((4 + (1 + 3), ), regressor.steps[0][1].coef_.shape)
		self.assertEqual((4 + (1 + 3), ), regressor.steps[1][1].coef_.shape)
		preds = estimator.predict(X)
		self.assertEqual((150, 2), preds.shape)

class SelectFirstClassifierTest(TestCase):

	def test_fit_predict(self):
		df = DataFrame([[-1, 0], [0, 0], [-1, -1], [1, 1], [-1, -1]], columns = ["X", "y"])
		X = df[["X"]]
		y = df["y"]
		classifier = clone(SelectFirstClassifier([
			("negative", DummyClassifier(strategy = "most_frequent"), "X[0] < 0"),
			("positive", DummyClassifier(strategy = "most_frequent"), "X[0] > 0"),
			("zero", DummyClassifier(strategy = "constant", constant = 0), str(True))
		]))
		params = classifier.get_params(deep = True)
		self.assertEqual("most_frequent", params["negative__strategy"])
		self.assertEqual("most_frequent", params["positive__strategy"])
		self.assertEqual("constant", params["zero__strategy"])
		self.assertEqual(0, params["zero__constant"])
		classifier.fit(X, y)
		preds = classifier.predict(X)
		self.assertEqual((5, ), preds.shape)
		self.assertEqual([-1, 0, -1, 1, -1], preds.tolist())
		pred_probs = classifier.predict_proba(X)
		self.assertEqual((5, 2), pred_probs.shape)

class SelectFirstRegressorTest(TestCase):
	pass