from pandas import DataFrame
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegression, SGDClassifier, SGDRegressor
from sklearn.svm import LinearSVC
from sklearn2pmml.ensemble import _checkLM, _checkLR, _step_params, SelectFirstEstimator

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

class SelectFirstEstimatorTest(TestCase):

	def test_fit_predict(self):
		df = DataFrame([[-1, 0], [0, 0], [-1, -1], [1, 1], [-1, -1]], columns = ["X", "y"])
		X = df[["X"]]
		y = df["y"]
		estimator = clone(SelectFirstEstimator([
			("negative", DummyClassifier(strategy = "most_frequent"), "X[0] < 0"),
			("positive", DummyClassifier(strategy = "most_frequent"), "X[0] > 0"),
			("zero", DummyClassifier(strategy = "constant", constant = 0), str(True))
		]))
		params = estimator.get_params(deep = True)
		self.assertEqual("most_frequent", params["negative__strategy"])
		self.assertEqual("most_frequent", params["positive__strategy"])
		self.assertEqual("constant", params["zero__strategy"])
		self.assertEqual(0, params["zero__constant"])
		estimator.fit(X, y)
		preds = estimator.predict(X)
		self.assertEqual([-1, 0, -1, 1, -1], preds.tolist())