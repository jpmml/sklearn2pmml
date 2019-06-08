from sklearn.base import clone, BaseEstimator
from sklearn.preprocessing import OneHotEncoder

class GBDTLR(BaseEstimator):

	def __init__(self, gbdt, lr):
		self.gbdt = gbdt
		self.lr = lr

	def _leaf_indices(self, X):
		id = self.gbdt_.apply(X)
		if id.ndim > 2:
			id = id[:, :, 0]
		return id

	def _encoded_leaf_indices(self, X):
		id = self._leaf_indices(X)
		idt = self.ohe_.transform(id)
		return idt

	def fit(self, X, y):
		self.gbdt_ = clone(self.gbdt)
		self.gbdt_.fit(X, y)
		id = self._leaf_indices(X)
		self.ohe_ = OneHotEncoder(categories = "auto")
		self.ohe_.fit(id)
		idt = self.ohe_.transform(id)
		self.lr_ = clone(self.lr)
		self.lr_.fit(idt, y)
		return self

	def predict(self, X):
		idt = self._encoded_leaf_indices(X)
		return self.lr_.predict(idt)

	def predict_proba(self, X):
		idt = self._encoded_leaf_indices(X)
		return self.lr_.predict_proba(idt)