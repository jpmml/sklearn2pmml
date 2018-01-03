import re
import sys
import unicodedata

punctuation = ""

for i in range(sys.maxunicode):
	c = i
	try:
		c = unichr(c)
	except:
		c = chr(c)
	if (unicodedata.category(c)).startswith("P"):
		punctuation += c

class Splitter:

	def __init__(self, separator_re = "\s+"):
		self.separator_re = separator_re

	def __getstate__(self):
		return self.separator_re

	def __setstate__(self, separator_re):
		self.separator_re = separator_re

	def __call__(self, text):
		tokens = re.split(self.separator_re, text)
		# Trim tokens by removing leading and trailing puncutation characters
		tokens = [token.strip(punctuation) for token in tokens]
		# Remove empty tokens
		tokens = [token for token in tokens if token]
		return tuple(tokens)
