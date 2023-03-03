from sklearn.base import BaseEstimator, RegressorMixin
from sklearn2pmml.util import eval_rows, to_expr_func, Expression

class ExpressionRegressor(BaseEstimator, RegressorMixin):

	def __init__(self, expr):
		if not isinstance(expr, Expression):
			raise TypeError()
		self.expr = expr

	def fit(self, X, y):
		return self

	def predict(self, X):
		func = to_expr_func(self.expr)
		return eval_rows(X, func, dtype = float)
