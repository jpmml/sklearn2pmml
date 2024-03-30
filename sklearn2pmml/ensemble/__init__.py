from sklearn.base import clone, BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.linear_model._base import LinearClassifierMixin, LinearModel, SparseCoefMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.metaestimators import _BaseComposition
from sklearn2pmml.util import check_predicate, eval_rows, fqn, to_expr_func

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

	@property
	def classes_(self):
		return self.lr_.classes_

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

class _BaseEnsemble(_BaseComposition):

	def __init__(self, steps, controller):
		for step in steps:
			if type(step) is not tuple:
				raise TypeError("Step is not a tuple")
			if len(step) != 3:
				raise TypeError("Step is not a three-element (name, estimator, predicate) tuple")
			name, estimator, predicate = step
			check_predicate(predicate)
		self.steps = steps
		if controller:
			if not hasattr(controller, "transform"):
				raise TypeError()
		self.controller = controller

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

	def _to_evaluation_dataset(self, X):
		if self.controller is not None:
			return self.controller.transform(X)
		return X

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

	def __init__(self, steps, controller = None, multioutput = True):
		super(EstimatorChain, self).__init__(steps, controller)
		self.multioutput = multioutput

	def fit(self, X, y, **fit_params):
		if len(y.shape) > 1:
			if len(self.steps) != y.shape[1]:
				raise ValueError()
			y = numpy.asarray(y)
		X_eval = self._to_evaluation_dataset(X)
		i = 0
		for name, estimator, predicate in self.steps:
			step_mask_func = to_expr_func(predicate)
			step_mask = eval_rows(X_eval, step_mask_func, dtype = bool)
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
		X_eval = self._to_evaluation_dataset(X)
		for name, estimator, predicate in self.steps:
			step_mask_func = to_expr_func(predicate)
			step_mask = eval_rows(X_eval, step_mask_func, dtype = bool)
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

	def __init__(self, steps, controller, eval_rows):
		super(SelectFirstEstimator, self).__init__(steps, controller)
		self.eval_rows = eval_rows

	def _eval_step_mask(self, X, predicate):
		step_func = to_expr_func(predicate)
		if self.eval_rows:
			return eval_rows(X, step_func, dtype = bool)
		else:
			return step_func(X)

	def fit(self, X, y, **fit_params):
		X_eval = self._to_evaluation_dataset(X)
		mask = numpy.full(X.shape[0], fill_value = False)
		for name, estimator, predicate in self.steps:
			step_mask = self._eval_step_mask(X_eval, predicate)
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
		X_eval = self._to_evaluation_dataset(X)
		mask = numpy.full(X.shape[0], fill_value = False)
		for name, estimator, predicate in self.steps:
			step_mask = self._eval_step_mask(X_eval, predicate)
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

	def __init__(self, steps, controller = None, eval_rows = True):
		super(SelectFirstRegressor, self).__init__(steps, controller, eval_rows)

class SelectFirstClassifier(SelectFirstEstimator, ClassifierMixin):

	def __init__(self, steps, controller = None, eval_rows = True):
		super(SelectFirstClassifier, self).__init__(steps, controller, eval_rows)

	def predict_proba(self, X):
		return self._predict(X, "predict_proba")
