from sklearn.base import TransformerMixin

class StatelessTransformerMixin(TransformerMixin):

	def __sklearn_is_fitted__(self):
		return True
