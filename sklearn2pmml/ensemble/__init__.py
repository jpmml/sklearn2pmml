from sklearn.base import clone, BaseEstimator, ClassifierMixin, RegressorMixin
try:
	from sklearn.linear_model import LinearRegression
	from sklearn.linear_model._base import LinearClassifierMixin, LinearModel, SparseCoefMixin
except ImportError:
	from sklearn.linear_model.base import LinearClassifierMixin, LinearModel, LinearRegression, SparseCoefMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.metaestimators import _BaseComposition
from sklearn2pmml.util import eval_rows

import numpy

def _class_name(x):
	return str(x.__class__)

def _checkGBDTRegressor(gbdt):
	if hasattr(gbdt, "apply"):
		return gbdt
	else:
		try:
			from lightgbm import LGBMRegressor
			if isinstance(gbdt, LGBMRegressor):
				return gbdt
		except ImportError:
			pass
	raise ValueError("GBDT class " + _class_name(gbdt) + " is not supported")

def _checkLM(lm):
	if isinstance(lm, (LinearModel, LinearRegression, SparseCoefMixin)):
		return lm
	raise ValueError("LM class " + _class_name(lm) + " is not supported")

def _checkGBDTClassifier(gbdt):
	if hasattr(gbdt, "apply"):
		return gbdt
	else:
		try:
			from lightgbm import LGBMClassifier
			if isinstance(gbdt, LGBMClassifier):
				return gbdt
		except ImportError:
			pass
	raise ValueError("GBDT class " + _class_name(gbdt) + " is not supported")

def _checkLR(lr):
	if isinstance(lr, LinearClassifierMixin):
		return lr
	raise ValueError("LR class " + _class_name(lr) + " is not supported")

def _step_params(name, params):
	prefix = name + "__"
	step_params = dict()
	for k, v in (params.copy()).items():
		if k.startswith(prefix):
			step_params[k[len(prefix):len(k)]] = v
			del params[k]
	return step_params

class GBDTEstimator(BaseEstimator):

	def __init__(self, gbdt):
		self.gbdt = gbdt

	def _leaf_indices(self, X):
		# Scikit-Learn
		# XGBoost
		if hasattr(self.gbdt_, "apply"):
			id = self.gbdt_.apply(X)
			if id.ndim > 2:
				id = id[:, :, 0]
		# LightGBM
		else:
			id = self.gbdt_.predict(X, pred_leaf = True)
		return id

	def _encoded_leaf_indices(self, X):
		id = self._leaf_indices(X)
		idt = self.ohe_.transform(id)
		return idt

class GBDTLMRegressor(GBDTEstimator, RegressorMixin):

	def __init__(self, gbdt, lm):
		super(GBDTLMRegressor, self).__init__(_checkGBDTRegressor(gbdt))
		self.lm = _checkLM(lm)

	def fit(self, X, y, **fit_params):
		self.gbdt_ = clone(self.gbdt)
		self.gbdt_.fit(X, y, **_step_params("gbdt", fit_params))
		id = self._leaf_indices(X)
		self.ohe_ = OneHotEncoder(categories = "auto")
		self.ohe_.fit(id)
		idt = self.ohe_.transform(id)
		self.lm_ = clone(self.lm)
		self.lm_.fit(idt, y, **_step_params("lm", fit_params))
		return self

	def predict(self, X):
		idt = self._encoded_leaf_indices(X)
		return self.lm_.predict(idt)

class GBDTLRClassifier(GBDTEstimator, ClassifierMixin):

	def __init__(self, gbdt, lr):
		super(GBDTLRClassifier, self).__init__(_checkGBDTClassifier(gbdt))
		self.lr = _checkLR(lr)

	def fit(self, X, y, **fit_params):
		self.gbdt_ = clone(self.gbdt)
		self.gbdt_.fit(X, y, **_step_params("gbdt", fit_params))
		id = self._leaf_indices(X)
		self.ohe_ = OneHotEncoder(categories = "auto")
		self.ohe_.fit(id)
		idt = self.ohe_.transform(id)
		self.lr_ = clone(self.lr)
		self.lr_.fit(idt, y, **_step_params("lr", fit_params))
		return self

	def predict(self, X):
		idt = self._encoded_leaf_indices(X)
		return self.lr_.predict(idt)

	def predict_proba(self, X):
		idt = self._encoded_leaf_indices(X)
		return self.lr_.predict_proba(idt)

class SelectFirstEstimator(_BaseComposition):

	def __init__(self, steps):
		for step in steps:
			if type(step) is not tuple:
				raise ValueError("Step is not a tuple")
			if len(step) != 3:
				raise ValueError("Step is not a three-element (name, estimator, predicate) tuple")
		self.steps = steps

	@property
	def _steps(self):
		return [(name, estimator) for name, estimator, predicate in self.steps]

	@_steps.setter
	def _steps(self, value):
		self.steps = [(name, estimator, predicate) for ((name, estimator), (_, _, predicate)) in zip(value, self.steps)]

	def get_params(self, deep = True):
		return self._get_params("_steps", deep = deep)

	def set_params(self, **kwargs):
		self._set_params("_steps", **kwargs)
		return self

	def fit(self, X, y, **fit_params):
		mask = numpy.zeros(X.shape[0], dtype = bool)
		for name, estimator, predicate in self.steps:
			step_mask = eval_rows(X, lambda X: eval(predicate), dtype = bool)
			step_mask[mask] = False
			if numpy.sum(step_mask) < 1:
				raise ValueError(predicate)
			estimator.fit(X[step_mask], y[step_mask], **_step_params(name, fit_params))
			mask = numpy.logical_or(mask, step_mask)
		return self

	def predict(self, X):
		result = numpy.empty((X.shape[0], ), dtype = object)
		mask = numpy.zeros(X.shape[0], dtype = bool)
		for name, estimator, predicate in self.steps:
			step_mask = eval_rows(X, lambda X: eval(predicate), dtype = bool)
			step_mask[mask] = False
			if numpy.sum(step_mask) < 1:
				continue
			step_result = estimator.predict(X[step_mask])
			result[step_mask.ravel()] = step_result
			mask = numpy.logical_or(mask, step_mask)
		return result

class SelectFirstRegressor(SelectFirstEstimator, RegressorMixin):

	def __init__(self, steps):
		super(SelectFirstRegressor, self).__init__(steps)

class SelectFirstClassifier(SelectFirstEstimator, ClassifierMixin):

	def __init__(self, steps):
		super(SelectFirstClassifier, self).__init__(steps)

	def predict_proba(self, X):
		result = None
		mask = numpy.zeros(X.shape[0], dtype = bool)
		for name, estimator, predicate in self.steps:
			step_mask = eval_rows(X, lambda X: eval(predicate), dtype = bool)
			step_mask[mask] = False
			if numpy.sum(step_mask) < 1:
				continue
			step_result = estimator.predict_proba(X[step_mask])
			if result is None:
				result = numpy.empty((X.shape[0], step_result.shape[1]), dtype = object)
			result[step_mask.ravel(), :] = step_result
			mask = numpy.logical_or(mask, step_mask)
		return result
