from scipy.special import expit, softmax
from sklearn.preprocessing import normalize
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn2pmml.util import check_expression, eval_rows, to_expr_func

import numpy

class ExpressionRegressor(BaseEstimator, RegressorMixin):

	def __init__(self, expr, normalization_method):
		self.expr = check_expression(expr)
		normalization_methods = ["none", "exp"]
		if normalization_method not in normalization_methods:
			raise ValueError("Normalization method {0} not in {1}".format(normalization_method, normalization_methods))
		self.normalization_method = normalization_method

	def fit(self, X, y):
		return self

	def decision_function(self, X):
		func = to_expr_func(self.expr)
		return eval_rows(X, func, dtype = float)

	def predict(self, X):
		y = self.decision_function(X)
		if self.normalization_method == "none":
			return y
		elif self.normalization_method == "exp":
			return numpy.exp(y)
		else:
			raise ValueError()

class ExpressionClassifier(BaseEstimator, ClassifierMixin):

	def __init__(self, class_exprs, normalization_method):
		if type(class_exprs) is not dict:
			raise TypeError()
		if not class_exprs:
			raise ValueError()
		for k, v in class_exprs.items():
			if k is None:
				raise ValueError()
			check_expression(v)
		self.class_exprs = class_exprs
		normalization_methods = ["none", "logit", "simplemax", "softmax"]
		if normalization_method not in normalization_methods:
			raise ValueError("Normalization method {0} not in {1}".format(normalization_method, normalization_methods))
		self.normalization_method = normalization_method

	def fit(self, X, y):
		self.classes_ = numpy.unique(y)
		n_classes = len(self.classes_)
		classes = set(self.class_exprs.keys())
		if n_classes == 2 and self.normalization_method in ["none", "logit"]:
			# One of the two classes
			if len(classes) != 1 or classes.intersection(self.classes_) != classes:
				raise ValueError()
		elif n_classes >= 2 and self.normalization_method in ["simplemax", "softmax"]:
			# All classes
			if len(classes) != n_classes or classes.intersection(self.classes_) != classes:
				raise ValueError()
		elif n_classes > 2 and self.normalization_method in ["none"]:
			# All classes, except for the last class
			if len(classes) != (n_classes - 1) or classes.intersection(self.classes_[:-1]) != classes:
				raise ValueError()
		else:
			raise ValueError()
		return self

	def decision_function(self, X):
		n_classes = len(self.classes_)
		y = None
		for clazz in self.classes_:
			try:
				class_expr = self.class_exprs[clazz]				
			except KeyError:
				continue
			class_func = to_expr_func(class_expr)
			class_y = eval_rows(X, class_func, dtype = float)
			if y is None:
				y = class_y
			else:
				y = numpy.vstack((y, class_y))
		if n_classes >= 2 and self.normalization_method in ["simplemax", "softmax"]:
			y = y.T.reshape(X.shape[0], -1)
		elif n_classes > 2 and self.normalization_method in ["none"]:
			y = y.T.reshape(X.shape[0], -1)
		return y

	def predict(self, X):
		y_proba = self.predict_proba(X)
		indices = numpy.argmax(y_proba, axis = 1)
		return numpy.take(self.classes_, indices, axis = 0)

	def predict_proba(self, X):
		n_classes = len(self.classes_)
		y = self.decision_function(X)
		if n_classes == 2 and self.normalization_method in ["none", "logit"]:
			if self.normalization_method == "none":
				proba = numpy.clip(y, 0, 1)
			elif self.normalization_method == "logit":
				proba = expit(y)
			else:
				raise ValueError()
			clazz = next(iter(self.class_exprs.keys()))
			if (self.classes_).tolist().index(clazz) == 1:
				return numpy.vstack((1 - proba, proba)).T
			else:
				return numpy.vstack((proba, 1 - proba)).T
		elif n_classes >= 2 and self.normalization_method in ["simplemax", "softmax"]:
			if self.normalization_method == "simplemax":
				return normalize(y, norm = "l1", axis = 1)
			elif self.normalization_method == "softmax":
				return softmax(y, axis = 1)
			else:
				raise ValueError()
		elif n_classes > 2 and self.normalization_method in ["none"]:
			y_rowsum = numpy.sum(y, axis = 1)
			return numpy.hstack((y, (1 - y_rowsum).reshape(-1, 1)))
		else:
			raise ValueError()
