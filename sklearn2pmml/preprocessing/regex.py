import warnings

def make_regex_engine(pattern):
	try:
		import pcre
		return pcre.compile(pattern)
	except ImportError:
		warnings.warn("Perl Compatible Regular Expressions (PCRE) library is not available, falling back to built-in Regular Expressions (RE) library. Transformation results might not be reproducible between Python and PMML environments when using more complex patterns", Warning)
		import re
		return re.compile(pattern)