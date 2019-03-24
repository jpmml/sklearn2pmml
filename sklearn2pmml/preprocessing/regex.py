from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import column_or_1d
from sklearn2pmml.util import eval_rows

import warnings

class RegExEngine(object):

	def __init__(self, prog):
		self.prog = prog

	def matches(self, input):
		return bool(self.prog.search(input))

	def replace(self, replacement, input):
		return self.prog.sub(replacement, input)

def _regex_engine(pattern):
	try:
		import pcre
		prog = pcre.compile(pattern)
	except ImportError:
		warnings.warn("Perl Compatible Regular Expression (PCRE) library is not available, falling back to built-in Regular Expression (RE) library. Transformation results might not be reproducible between Python and PMML environments when using more complex patterns", Warning)
		import re
		prog = re.compile(pattern)
	return RegExEngine(prog)

class MatchesTransformer(BaseEstimator, TransformerMixin):

	def __init__(self, pattern):
		self.pattern = pattern

	def fit(self, X, y = None):
		X = column_or_1d(X, warn = True)
		return self

	def transform(self, X):
		X = column_or_1d(X, warn = True)
		engine = _regex_engine(self.pattern)
		func = lambda x: engine.matches(x)
		return eval_rows(X, func)

class ReplaceTransformer(BaseEstimator, TransformerMixin):

	def __init__(self, pattern, replacement):
		self.pattern = pattern
		self.replacement = replacement

	def fit(self, X, y = None):
		X = column_or_1d(X, warn = True)
		return self

	def transform(self, X):
		X = column_or_1d(X, warn = True)
		engine = _regex_engine(self.pattern)
		func = lambda x: engine.replace(self.replacement, x)
		return eval_rows(X, func)