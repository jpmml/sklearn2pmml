from sklearn.base import clone, BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model.base import SparseCoefMixin, LinearClassifierMixin, LinearModel, LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn2pmml.util import eval_rows

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
	if isinstance(lm, LinearModel) or isinstance(lm, LinearRegression) or isinstance(lm, SparseCoefMixin):
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

class SelectFirstEstimator(BaseEstimator):

	def __init__(self, steps):
		for step in steps:
			if type(step) is not tuple:
				raise ValueError("Step is not a tuple")
			if len(step) != 2:
				raise ValueError("Step is not a two-element (predicate, estimator) tuple")
		self.steps = steps

	def _eval_row(self, X):
		for step in self.steps:
			predicate = step[0]
			estimator = step[1]
			if eval(predicate):
				Xt = X.reshape(1, -1)
				y = estimator.predict(Xt)
				return y.item()
		return None

	def fit(self, X, y, **fit_params):
		raise NotImplementedError()

	def predict(self, X):
		func = lambda x: self._eval_row(x)
		return eval_rows(X, func)