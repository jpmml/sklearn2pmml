from scipy.interpolate import BSpline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import column_or_1d

class BSplineTransformer(BaseEstimator, TransformerMixin):

	def __init__(self, bspline):
		if not isinstance(bspline, BSpline):
			raise ValueError("Spline is not an instance of " + BSpline.__name__)
		self.bspline = bspline

	def fit(self, X, y = None):
		X = column_or_1d(X, warn = True)
		return self

	def transform(self, X):
		X = column_or_1d(X, warn = True)
		return self.bspline(X)
