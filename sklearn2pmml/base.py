from sklearn.base import TransformerMixin

class StatelessEstimatorMixin:

	def __sklearn_is_fitted__(self):
		return True

class StatelessTransformerMixin(TransformerMixin):

	def __sklearn_is_fitted__(self):
		return True
