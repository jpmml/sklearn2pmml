import warnings

class RegExEngine(object):

	def __init__(self, pattern):
		self.pattern = pattern

	def matches(self, x):
		raise NotImplementedError()

	def replace(self, replacement, x):
		raise NotImplementedError()

class REEngine(RegExEngine):

	def __init__(self, pattern):
		import re

		super(REEngine, self).__init__(pattern)
		self.pattern_ = re.compile(pattern)

	def matches(self, x):
		return self.pattern_.search(x)

	def replace(self, replacement, x):
		return self.pattern_.sub(replacement, x)

class PCREEngine(RegExEngine):

	def __init__(self, pattern):
		import pcre

		super(PCREEngine, self).__init__(pattern)
		self.pattern_ = pcre.compile(pattern)

	def matches(self, x):
		return self.pattern_.search(x)

	def replace(self, replacement, x):
		return self.pattern_.sub(replacement, x)

def make_regex_engine(pattern):
	try:
		return PCREEngine(pattern)
	except ImportError:
		warnings.warn("Perl Compatible Regular Expressions (PCRE) library is not available, falling back to built-in Regular Expressions (RE) library. Transformation results might not be reproducible between Python and PMML environments when using more complex patterns", Warning)
		return REEngine(pattern)