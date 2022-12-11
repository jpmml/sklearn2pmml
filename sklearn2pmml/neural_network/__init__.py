from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.neural_network import MLPRegressor

class MLPTransformer(BaseEstimator, TransformerMixin):

	def __init__(self, mlp, transformer_output_layer = -1):
		if not isinstance(mlp, MLPRegressor):
			raise TypeError("The mlp object is not an instance of {0}".format(MLPRegressor.__name__))
		self.mlp = mlp
		self.transformer_output_layer = transformer_output_layer

	def fit(self, X, y = None):
		self.mlp_ = clone(self.mlp)
		self.mlp_.fit(X, X)
		if self.transformer_output_layer != -1:
			self.mlp_.n_layers_ = self.transformer_output_layer + 1
			self.mlp_.n_outputs_ = self.mlp_.hidden_layer_sizes[self.transformer_output_layer - 1]
		return self

	def transform(self, X):
		return self.mlp_.predict(X)