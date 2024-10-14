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

def make_regex_engine(pattern, re_flavour):
	if re_flavour is None:
		try:
			import pcre2

			re_flavour = "pcre2"
		except ImportError:
			warnings.warn("Perl Compatible Regular Expressions (PCRE) library is not available, falling back to built-in Regular Expressions (RE) library. Transformation results might not be reproducible between Python and PMML environments when using more complex patterns", Warning)
			re_flavour = "re"

	if re_flavour == "pcre":
		return PCREEngine(pattern)
	elif re_flavour == "pcre2":
		return PCRE2Engine(pattern)
	elif re_flavour == "re":
		return REEngine(pattern)
	else:
		re_flavours = ["pcre", "pcre2", "re"]
		raise ValueError("Regular Expressions flavour {0} not in {1}".format(re_flavour, re_flavours))
