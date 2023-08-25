from sklearn.base import clone, BaseEstimator, ClassifierMixin, RegressorMixin
try:
	from sklearn.linear_model import LinearRegression
	from sklearn.linear_model._base import LinearClassifierMixin, LinearModel, SparseCoefMixin
except ImportError:
	from sklearn.linear_model.base import LinearClassifierMixin, LinearModel, LinearRegression, SparseCoefMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.metaestimators import _BaseComposition
from sklearn2pmml import _is_ordinal
from sklearn2pmml.util import eval_expr_rows, fqn, Predicate

import copy
import numpy

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
	raise TypeError("GBDT object (class {0}) is not supported".format(fqn(gbdt)))

def _checkLM(lm):
	if isinstance(lm, (LinearModel, LinearRegression, SparseCoefMixin)):
		return lm
	raise TypeError("LM object (class {0}) is not supported".format(fqn(lm)))

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
	raise TypeError("GBDT object (class {0}) is not supported".format(fqn(gbdt)))

def _checkLR(lr):
	if isinstance(lr, LinearClassifierMixin):
		return lr
	raise TypeError("LR object (class {0}) is not supported".format(fqn(lr)))

def _extract_step_params(name, params):
	prefix = name + "__"
	step_params = dict()
	for k, v in (params.copy()).items():
		if k.startswith(prefix):
			step_params[k[len(prefix):len(k)]] = v
			del params[k]
	return step_params

def _mask_params(params, mask):
	masked_params = dict()
	for k, v in params.items():
		if isinstance(v, numpy.ndarray):
			if v.shape[0] != mask.shape[0]:
				raise ValueError()
			v = v[mask]
		masked_params[k] = v
	return masked_params

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
		self.gbdt_.fit(X, y, **_extract_step_params("gbdt", fit_params))
		id = self._leaf_indices(X)
		self.ohe_ = OneHotEncoder(categories = "auto")
		self.ohe_.fit(id)
		idt = self.ohe_.transform(id)
		self.lm_ = clone(self.lm)
		self.lm_.fit(idt, y, **_extract_step_params("lm", fit_params))
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
		self.gbdt_.fit(X, y, **_extract_step_params("gbdt", fit_params))
		id = self._leaf_indices(X)
		self.ohe_ = OneHotEncoder(categories = "auto")
		self.ohe_.fit(id)
		idt = self.ohe_.transform(id)
		self.lr_ = clone(self.lr)
		self.lr_.fit(idt, y, **_extract_step_params("lr", fit_params))
		return self

	def predict(self, X):
		idt = self._encoded_leaf_indices(X)
		return self.lr_.predict(idt)

	def predict_proba(self, X):
		idt = self._encoded_leaf_indices(X)
		return self.lr_.predict_proba(idt)

def _codes(y):
	if hasattr(y, "cat"):
		return (y.cat).codes.values
	else:
		return y.codes

class OrdinalClassifier(BaseEstimator, ClassifierMixin):

	def __init__(self, estimator):
		self.estimator = estimator

	def fit(self, X, y, **fit_params):
		dtype = y.dtype
		if not _is_ordinal(dtype):
			raise TypeError()
		y_codes = _codes(y)
		self.classes_ = numpy.asarray(dtype.categories)
		self.estimators_ = []
		n_estimators = len(self.classes_) - 1
		for i in range(n_estimators):
			y_bin = (y_codes > i).astype(int)
			estimator = clone(self.estimator)
			estimator.fit(X, y_bin, **fit_params)
			self.estimators_.append(estimator)
		return self

	def predict(self, X, **predict_params):
		proba = self.predict_proba(X, **predict_params)
		indices = numpy.argmax(proba, axis = 1)
		return numpy.take(self.classes_, indices)

	def predict_proba(self, X, **predict_proba_params):
		n_estimators = len(self.classes_) - 1
		proba = []
		for i in range(n_estimators):
			proba.append(self.estimators_[i].predict_proba(X, **predict_proba_params)[:, 1])
		result = []
		for i in range(n_estimators + 1):
			if i == 0:
				result.append(1 - proba[0])
			elif i == n_estimators:
				result.append(proba[-1])
			else:
				result.append(proba[i - 1] - proba[i])
		return numpy.asarray(result).T

class _BaseEnsemble(_BaseComposition):

	def __init__(self, steps):
		for step in steps:
			if type(step) is not tuple:
				raise TypeError("Step is not a tuple")
			if len(step) != 3:
				raise TypeError("Step is not a three-element (name, estimator, predicate) tuple")
			name, estimator, predicate = step
			if not isinstance(predicate, (str, Predicate)):
				raise TypeError()
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

def _to_sparse(X, step_mask, step_result):
	# Make array
	if len(step_result.shape) == 1:
		result = numpy.empty((X.shape[0], ), dtype = object)
	else:
		result = numpy.empty((X.shape[0], step_result.shape[1]), dtype = object)
	# Fill array
	if len(step_result.shape) == 1:
		result[step_mask.ravel()] = step_result
	else:
		result[step_mask.ravel(), :] = step_result
	return result

class Link(BaseEstimator):

	def __init__(self, estimator, augment_funcs, prefit = False):
		self.estimator = estimator
		self.augment_funcs = augment_funcs
		self.prefit = prefit
		if prefit:
			self.estimator_ = copy.deepcopy(self.estimator)

	def fit(self, X, y, **fit_params):
		self.estimator_ = clone(self.estimator)
		self.estimator_.fit(X, y, **fit_params)
		return self

	def predict(self, X):
		return self.estimator_.predict(X)

	def predict_proba(self, X):
		return self.estimator_.predict_proba(X)

	def augment(self, X):
		Y = None
		for augment_func in self.augment_funcs:
			yt = getattr(self.estimator_, augment_func)(X)
			yt = yt.reshape(X.shape[0], -1)
			if Y is None:
				Y = yt
			else:
				Y = numpy.hstack((Y, yt))
		if Y is None:
			return X
		else:
			return numpy.hstack((X, Y))

class EstimatorChain(_BaseEnsemble):

	def __init__(self, steps, multioutput = True):
		super(EstimatorChain, self).__init__(steps)
		self.multioutput = multioutput

	def fit(self, X, y, **fit_params):
		if len(y.shape) > 1:
			if len(self.steps) != y.shape[1]:
				raise ValueError()
			y = numpy.asarray(y)
		i = 0
		for name, estimator, predicate in self.steps:
			step_mask = eval_expr_rows(X, predicate, dtype = bool)
			if numpy.sum(step_mask) < 1:
				raise ValueError(predicate)
			step_X = X[step_mask]
			if len(y.shape) == 1:
				step_y = y[step_mask]
			elif len(y.shape) == 2:
				step_y = y[step_mask, i]
			else:
				raise ValueError()
			step_fit_params = _mask_params(_extract_step_params(name, fit_params), step_mask)
			estimator.fit(step_X, step_y, **step_fit_params)
			if isinstance(estimator, Link):
				X = estimator.augment(X)
			i += 1
		return self

	def _predict(self, X, predict_method):
		result = None
		for name, estimator, predicate in self.steps:
			step_mask = eval_expr_rows(X, predicate, dtype = bool)
			if numpy.sum(step_mask) < 1:
				continue
			step_X = X[step_mask]
			step_result = getattr(estimator, predict_method)(step_X)
			step_result = _to_sparse(X, step_mask, step_result)
			if self.multioutput:
				step_result = step_result.reshape(X.shape[0], -1)
			if result is None:
				result = step_result
			else:
				if self.multioutput:
					result = numpy.hstack((result, step_result))
				else:
					if step_result.shape != result.shape:
						raise ValueError()
					result[step_mask] = step_result[step_mask]
			if isinstance(estimator, Link):
				X = estimator.augment(X)
		return result

	def predict(self, X):
		return self._predict(X, "predict")

	def predict_proba(self, X):
		return self._predict(X, "predict_proba")

class SelectFirstEstimator(_BaseEnsemble):

	def __init__(self, steps):
		super(SelectFirstEstimator, self).__init__(steps)

	def fit(self, X, y, **fit_params):
		mask = numpy.zeros(X.shape[0], dtype = bool)
		for name, estimator, predicate in self.steps:
			step_mask = eval_expr_rows(X, predicate, dtype = bool)
			step_mask[mask] = False
			if numpy.sum(step_mask) < 1:
				raise ValueError(predicate)
			step_X = X[step_mask]
			step_y = y[step_mask]
			step_fit_params = _mask_params(_extract_step_params(name, fit_params), step_mask)
			estimator.fit(step_X, step_y, **step_fit_params)
			mask = numpy.logical_or(mask, step_mask)
		return self

	def _predict(self, X, predict_method):
		result = None
		mask = numpy.zeros(X.shape[0], dtype = bool)
		for name, estimator, predicate in self.steps:
			step_mask = eval_expr_rows(X, predicate, dtype = bool)
			step_mask[mask] = False
			if numpy.sum(step_mask) < 1:
				continue
			step_X = X[step_mask]
			step_result = getattr(estimator, predict_method)(step_X)
			step_result = _to_sparse(X, step_mask, step_result)
			if result is None:
				result = step_result
			else:
				result[step_mask] = step_result[step_mask]
			mask = numpy.logical_or(mask, step_mask)
		return result

	def apply(self, X):
		return self._predict(X, "apply")

	def predict(self, X):
		return self._predict(X, "predict")

class SelectFirstRegressor(SelectFirstEstimator, RegressorMixin):

	def __init__(self, steps):
		super(SelectFirstRegressor, self).__init__(steps)

class SelectFirstClassifier(SelectFirstEstimator, ClassifierMixin):

	def __init__(self, steps):
		super(SelectFirstClassifier, self).__init__(steps)

	def predict_proba(self, X):
		return self._predict(X, "predict_proba")
