from scipy.interpolate import BSpline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn2pmml.util import ensure_1d

class BSplineTransformer(BaseEstimator, TransformerMixin):

	def __init__(self, bspline):
		if not isinstance(bspline, BSpline):
			raise ValueError("Spline is not an instance of " + BSpline.__name__)
		self.bspline = bspline

	def fit(self, X, y = None):
		X = ensure_1d(X)
		return self

	def transform(self, X):
		X = ensure_1d(X)
		return self.bspline(X)
