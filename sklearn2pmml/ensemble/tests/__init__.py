from pandas import CategoricalDtype, DataFrame, Series
from sklearn.base import clone
from sklearn.datasets import load_diabetes, load_iris
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegression, SGDClassifier, SGDRegressor
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn2pmml.ensemble import _checkLM, _checkLR, _extract_step_params, _mask_params, EstimatorChain, Link, OrdinalClassifier, SelectFirstClassifier, SelectFirstRegressor
from unittest import TestCase

import numpy

class GBDTLRTest(TestCase):

	def test_lm(self):
		_checkLM(ElasticNet())
		_checkLM(LinearRegression())
		_checkLM(SGDRegressor())

	def test_lr(self):
		_checkLR(LinearSVC())
		_checkLR(LogisticRegression())
		_checkLR(SGDClassifier())

class EstimatorChainTest(TestCase):

	def test_fit_predict(self):
		df = DataFrame([[-1, 0], [0, 0], [-1, -1], [1, 1], [-1, -1]], columns = ["X", "y"])
		X = df[["X"]]
		y = df["y"]
		steps = [
			("negative", DummyClassifier(strategy = "most_frequent"), "X[0] < 0"), # binary
			("not_negative", DummyClassifier(strategy = "most_frequent"), "X[0] >= 0"), # binary
			("any", DummyClassifier(strategy = "most_frequent"), str(True)) # multiclass
		]
		estimator = EstimatorChain(steps, multioutput = True)
		params = estimator.get_params(deep = True)
		self.assertEqual("most_frequent", params["negative__strategy"])
		self.assertEqual("most_frequent", params["not_negative__strategy"])
		self.assertEqual("most_frequent", params["any__strategy"])
		estimator.fit(X, y)
		pred = estimator.predict(X)
		self.assertEqual((5, 3), pred.shape)
		self.assertEqual([-1, None, -1, None, -1], pred[:, 0].tolist())
		self.assertEqual([None, 0, None, 0, None], pred[:, 1].tolist())
		self.assertEqual([-1, -1, -1, -1, -1], pred[:, 2].tolist())
		pred_proba = estimator.predict_proba(X)
		self.assertEqual((5, 2 + 2 + 3), pred_proba.shape)
		self.assertEqual([[1, 0], [None, None], [1, 0], [None, None], [1, 0]], pred_proba[:, 0:2].tolist())
		self.assertEqual([[None, None], [1, 0], [None, None], [1, 0], [None, None]], pred_proba[:, 2:4].tolist())
		self.assertEqual([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]], pred_proba[:, 4:7].tolist())

		estimator = EstimatorChain(steps, multioutput = False)
		estimator.fit(X, y)
		pred = estimator.predict(X)
		self.assertEqual((5, ), pred.shape)
		self.assertEqual([-1, -1, -1, -1, -1], pred.tolist())
		with self.assertRaises(ValueError):
			estimator.predict_proba(X)
		estimator = EstimatorChain(steps[0:2], multioutput = False)
		estimator.fit(X, y)
		pred = estimator.predict(X)
		self.assertEqual((5, ), pred.shape)
		self.assertEqual([-1, 0, -1, 0, -1], pred.tolist())
		pred_proba = estimator.predict_proba(X)
		self.assertEqual((5, 2), pred_proba.shape)
		# XXX: non-sensical, because the two binary classifiers have different class labels ("negative/not negative" vs "not-negative/not not-negative")
		self.assertEqual([[1, 0], [1, 0], [1, 0], [1, 0], [1, 0]], pred_proba.tolist())

	def test_complex_fit_predict(self):
		X, y = load_iris(return_X_y = True)
		classifier = Link(DecisionTreeClassifier(max_depth = 2, random_state = 13), augment_funcs = ["predict", "predict_proba"])
		regressor = SelectFirstRegressor([
			("not_setosa", LinearRegression(), "X[-3] < 0.5"),
			("setosa", LinearRegression(), "X[-3] >= 0.5")
		])
		estimator = EstimatorChain([
			("classifier", classifier, str(True)),
			("regressor", regressor, str(True))
		])
		estimator.fit(X, y)
		if hasattr(classifier.estimator_, "n_features_"):
			self.assertEqual(4, classifier.estimator_.n_features_)
		else:
			self.assertEqual(4, classifier.estimator_.n_features_in_)
		self.assertEqual((4 + (1 + 3), ), regressor.steps[0][1].coef_.shape)
		self.assertEqual((4 + (1 + 3), ), regressor.steps[1][1].coef_.shape)
		pred = estimator.predict(X)
		self.assertEqual((150, 2), pred.shape)
		with self.assertRaises(AttributeError):
			estimator.predict_proba(X)

class OrdinalClassifierTest(TestCase):

	def test_fit_predict(self):
		diabetes_X, diabetes_y = load_diabetes(return_X_y = True)
		discretizer = KBinsDiscretizer(n_bins = 5, encode = "ordinal", strategy = "kmeans")
		diabetes_y = discretizer.fit_transform(diabetes_y.reshape((-1, 1))).astype(int)
		diabetes_y = numpy.vectorize(lambda x:"c{}".format(x + 1))(diabetes_y)
		diabetes_y = Series(diabetes_y.ravel(), dtype = CategoricalDtype(["c1", "c2", "c3", "c4", "c5"], ordered = True))
		classifier = OrdinalClassifier(LogisticRegression())
		classifier.fit(diabetes_X, diabetes_y)
		pred = classifier.predict(diabetes_X)
		self.assertEqual((442, ), pred.shape)
		pred_proba = classifier.predict_proba(diabetes_X)
		self.assertEqual((442, 5), pred_proba.shape)

class SelectFirstClassifierTest(TestCase):

	def test_fit_predict(self):
		df = DataFrame([[-1, 0], [0, 0], [-1, -1], [1, 1], [-1, -1]], columns = ["X", "y"])
		X = df[["X"]]
		y = df["y"]
		classifier = clone(SelectFirstClassifier([
			("negative", DummyClassifier(strategy = "most_frequent"), "X[0] < 0"),
			("positive", DummyClassifier(strategy = "most_frequent"), "X[0] > 0"),
			("zero", DummyClassifier(strategy = "constant", constant = 0), str(True))
		], eval_rows = True))
		params = classifier.get_params(deep = True)
		self.assertEqual("most_frequent", params["negative__strategy"])
		self.assertEqual("most_frequent", params["positive__strategy"])
		self.assertEqual("constant", params["zero__strategy"])
		self.assertEqual(0, params["zero__constant"])
		classifier.fit(X, y)
		pred = classifier.predict(X)
		self.assertEqual((5, ), pred.shape)
		self.assertEqual([-1, 0, -1, 1, -1], pred.tolist())
		pred_proba = classifier.predict_proba(X)
		self.assertEqual((5, 2), pred_proba.shape)
		X = X.values
		classifier = SelectFirstClassifier([
			("negative", DummyClassifier(strategy = "most_frequent"), "X[:, 0] < 0"),
			("positive", DummyClassifier(strategy = "most_frequent"), "X[:, 0] > 0"),
			("zero", DummyClassifier(strategy = "constant", constant = 0), "X[:, 0] == 0")
		], eval_rows = False)
		classifier.fit(X, y)
		pred = classifier.predict(X)
		self.assertEqual([-1, 0, -1, 1, -1], pred.tolist())

class SelectFirstRegressorTest(TestCase):
	pass

class FunctionTest(TestCase):

	def test_extract_step_params(self):
		params = {
			"gbdt__first" : 1,
			"lr__first" : 1.0,
			"gbdt__second" : 2,
			"any__any" : None
		}
		gbdt_params = _extract_step_params("gbdt", params)
		self.assertEqual({"first" : 1, "second" : 2}, gbdt_params)
		self.assertEqual({"lr__first" : 1.0, "any__any" : None}, params)
		lr_params = _extract_step_params("lr", params)
		self.assertEqual({"first" : 1.0}, lr_params)
		self.assertEqual({"any__any" : None}, params)

	def test_mask_params(self):
		params = {
			"first" : numpy.asarray([[0], [1], [1], [2], [0]]),
			"second" : numpy.asarray([[False, "A"], [False, "B"], [True, "C"], [False, "D"], [True, "E"]], dtype = object),
			"any" : "any"
		}
		mask = numpy.full((5, ), True)
		masked_params = _mask_params(params, mask)
		self.assertTrue(params["first"].tolist(), masked_params["first"].tolist())
		self.assertTrue(params["second"].tolist(), masked_params["second"].tolist())
		self.assertTrue(params["any"], masked_params["any"])
		mask = numpy.asarray([False, True, False, False, True], dtype = bool)
		masked_params = _mask_params(params, mask)
		self.assertEqual([[1], [0]], masked_params["first"].tolist())
		self.assertEqual([[False, "B"], [True, "E"]], masked_params["second"].tolist())
		self.assertEqual(params["any"], masked_params["any"])
		mask = numpy.full((5, ), False)
		masked_params = _mask_params(params, mask)
		self.assertEqual([], masked_params["first"].tolist())
		self.assertEqual([], masked_params["second"].tolist())
		self.assertEqual(params["any"], masked_params["any"])
