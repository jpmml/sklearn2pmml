from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from statsmodels.base.model import Model

import numpy

class StatsModelsEstimator(BaseEstimator):

	def __init__(self, model_class, **init_params):
		if not isinstance(model_class, type):
			raise TypeError("The model class object is not a Python class")
		if not issubclass(model_class, Model):
			raise TypeError("The model class is not a subclass of {}".format(Model.__name__))
		self.model_class = model_class
		self.init_params = init_params

	def fit(self, X, y, **fit_params):
		self.model_ = self.model_class(endog = y, exog = X, **self.init_params)
		self.result_ = self.model_.fit(**fit_params)
		return self

class StatsModelsClassifier(StatsModelsEstimator, ClassifierMixin):

	def __init__(self, model_class, **init_params):
		super(StatsModelsClassifier, self).__init__(model_class, **init_params)

	def fit(self, X, y, **fit_params):
		classes, y_encoded = numpy.unique(y, return_inverse = True)
		self.classes_ = classes
		super(StatsModelsClassifier, self).fit(X = X, y = y_encoded, **fit_params)
		return self

	def predict(self, X, **predict_params):
		proba = self.predict_proba(X, **predict_params)
		indices = numpy.argmax(proba, axis = 1)
		return numpy.take(self.classes_, indices)

	def predict_proba(self, X, **predict_proba_params):
		proba = self.result_.predict(X, **predict_proba_params)
		if proba.ndim == 1:
			proba = numpy.vstack((1 - proba, proba)).T
		return proba

class StatsModelsRegressor(StatsModelsEstimator, RegressorMixin):

	def __init__(self, model_class, **init_params):
		super(StatsModelsRegressor, self).__init__(model_class, **init_params)

	def predict(self, X, **predict_params):
		return self.result_.predict(exog = X, **predict_params)
