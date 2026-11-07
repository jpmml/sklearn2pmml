from sklearn.base import TransformerMixin
try:
	# SkLearn 1.6.0+
	from sklearn.utils.validation import _check_feature_names, _check_n_features
except ImportError:
	_check_feature_names = None
	_check_n_features = None

import numpy

class FeaturesInMixin:

	def _check_features_in(self, X, reset):
		if len(numpy.shape(X)) == 1:
			if reset:
				self.n_features_in_ = 1
			return
		# SkLearn 1.5.X
		if _check_n_features is None:
			self._check_n_features(X, reset = reset)
			self._check_feature_names(X, reset = reset)
		# SkLearn 1.6.0+
		else:
			_check_n_features(self, X, reset = reset)
			_check_feature_names(self, X, reset = reset)

class StatelessEstimatorMixin:

	def __sklearn_is_fitted__(self):
		return True

class StatelessTransformerMixin(TransformerMixin):

	def __sklearn_is_fitted__(self):
		return True
