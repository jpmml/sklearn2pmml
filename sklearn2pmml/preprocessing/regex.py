import warnings

class RegExEngine(object):

	def __init__(self, pattern):
		self.pattern = pattern

	def matches(self, x):
		raise NotImplementedError()

	def replace(self, replacement, x):
		raise NotImplementedError()

class PCREEngine(RegExEngine):

	def __init__(self, pattern):
		import pcre

		super(PCREEngine, self).__init__(pattern)
		self.pattern_ = pcre.compile(pattern)

	def matches(self, x):
		return self.pattern_.search(x)

	def replace(self, replacement, x):
		return self.pattern_.sub(replacement, x)

class PCRE2Engine(RegExEngine):

	def __init__(self, pattern):
		import pcre2

		super(PCRE2Engine, self).__init__(pattern)
		self.pattern_ = pcre2.compile(pattern)

	def matches(self, x):
		scanner = self.pattern_.scan(x)
		try:
			scanner.__next__()
			return True
		except StopIteration:
			return False

	def replace(self, replacement, x):
		return self.pattern_.substitute(replacement, x)

class REEngine(RegExEngine):

	def __init__(self, pattern):
		import re

		super(REEngine, self).__init__(pattern)
		self.pattern_ = re.compile(pattern)

		warnings.warn("Using Python's built-in Regular Expressions (RE) engine instead of Perl Compatible Regular Expressions (PCRE) engine. Transformation results might not be reproducible between Python and PMML environments when using more complex patterns", Warning)

	def matches(self, x):
		return self.pattern_.search(x)

	def replace(self, replacement, x):
		return self.pattern_.sub(replacement, x)

def make_regex_engine(pattern, re_flavour):
	if re_flavour == "pcre":
		return PCREEngine(pattern)
	elif re_flavour == "pcre2":
		return PCRE2Engine(pattern)
	elif re_flavour == "re":
		return REEngine(pattern)
	else:
		raise ValueError(re_flavour)
