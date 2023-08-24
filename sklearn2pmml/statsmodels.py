from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn2pmml import _is_ordinal
from statsmodels.base.model import Model
from statsmodels.tools import add_constant

import numpy

class StatsModelsEstimator(BaseEstimator):

	def __init__(self, model_class, fit_intercept = True, **init_params):
		if not isinstance(model_class, type):
			raise TypeError("The model class object is not a Python class")
		if not issubclass(model_class, Model):
			raise TypeError("The model class is not a subclass of {}".format(Model.__name__))
		self.model_class = model_class
		self.fit_intercept = fit_intercept
		self.init_params = init_params

	def fit(self, X, y, **fit_params):
		if self.fit_intercept:
			X = add_constant(X, has_constant = "add")
		self.model_ = self.model_class(endog = y, exog = X, **self.init_params)
		fit_method = fit_params.pop("fit_method", "fit")
		if (fit_method != "fit") and (not fit_method.startswith("fit_")):
			raise ValueError()
		self.results_ = getattr(self.model_, fit_method)(**fit_params)
		return self

	def remove_data(self):
		self.results_.remove_data()

class StatsModelsClassifier(StatsModelsEstimator, ClassifierMixin):

	def __init__(self, model_class, fit_intercept = True, **init_params):
		super(StatsModelsClassifier, self).__init__(model_class = model_class, fit_intercept = fit_intercept, **init_params)

	def regression_table_(self):
		params = self.results_.params.T
		if len(params.shape) == 1:
			params = params.reshape((1, -1))
		return params

	@property
	def coef_(self):
		reg_table = self.regression_table_()
		if self.fit_intercept:
			return reg_table[:, 1:]
		return reg_table

	@property
	def intercept_(self):
		reg_table = self.regression_table_()
		if self.fit_intercept:
			return reg_table[:, 0]
		return numpy.zeros((reg_table.shape[0], ), dtype = float)

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
		if self.fit_intercept:
			X = add_constant(X, has_constant = "add")
		proba = self.results_.predict(X, **predict_proba_params)
		if proba.ndim == 1:
			proba = numpy.vstack((1 - proba, proba)).T
		return proba

class StatsModelsOrdinalClassifier(StatsModelsEstimator):

	def __init__(self, model_class, **init_params):
		super(StatsModelsOrdinalClassifier, self).__init__(model_class = model_class, fit_intercept = False, **init_params)

	def regression_table_(self):
		params = self.results_.params
		coef_params = params[0:-(len(self.classes_) - 1)]
		return coef_params.T

	@property
	def coef_(self):
		reg_table = self.regression_table_()
		return reg_table

	@property
	def intercept_(self):
		if hasattr(self.results_, "offset"):
			return self.results_.offset
		return 0.0

	@property
	def threshold_(self):
		params = self.results_.params
		th_params = params[-(len(self.classes_) - 1):]
		return th_params.T

	def fit(self, X, y, **fit_params):
		dtype = y.dtype
		if not _is_ordinal(dtype):
			raise TypeError()
		self.classes_ = numpy.asarray(dtype.categories)
		super(StatsModelsOrdinalClassifier, self).fit(X = X, y = y, **fit_params)
		return self

	def predict(self, X, **predict_params):
		proba = self.predict_proba(X, **predict_params)
		indices = numpy.argmax(proba, axis = 1)
		return numpy.take(self.classes_, indices)

	def predict_proba(self, X, **predict_proba_params):
		proba = self.results_.predict(X, **predict_proba_params)
		return proba

class StatsModelsRegressor(StatsModelsEstimator, RegressorMixin):

	def __init__(self, model_class, fit_intercept = True, **init_params):
		super(StatsModelsRegressor, self).__init__(model_class = model_class, fit_intercept = fit_intercept, **init_params)

	def regression_table_(self):
		return self.results_.params.T

	@property
	def coef_(self):
		reg_table = self.regression_table_()
		if self.fit_intercept:
			reg_table = reg_table[1:]
		return reg_table

	@property
	def intercept_(self):
		reg_table = self.regression_table_()
		if self.fit_intercept:
			return reg_table[0]
		return 0.0

	def predict(self, X, **predict_params):
		if self.fit_intercept:
			X = add_constant(X, has_constant = "add")
		return self.results_.predict(exog = X, **predict_params)
