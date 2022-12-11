from optbinning import OptimalBinning
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn2pmml.util import ensure_1d

class OptimalBinningWrapper(BaseEstimator, TransformerMixin):

	def __init__(self, optimal_binning, metric, prefit = False):
		if not isinstance(optimal_binning, OptimalBinning):
			raise TypeError("The optimal binning object is not an instance of {0}".format(OptimalBinning.__name__))
		self.optimal_binning = optimal_binning
		self.metric = metric
		self.prefit = prefit
		if self.prefit:
			self.optimal_binning_ = clone(self.optimal_binning)

	def fit(self, X, y = None, **fit_params):
		X = ensure_1d(X)
		self.optimal_binning_ = clone(self.optimal_binning)
		return self.optimal_binning_.fit(X, y, **fit_params)

	def transform(self, X):
		X = ensure_1d(X)
		return self.optimal_binning_.transform(X, metric = self.metric)
