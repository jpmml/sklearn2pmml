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

class Matcher:

	def __init__(self, word_re = "\w+"):
		self.word_re = word_re

	def __getstate__(self):
		return self.word_re

	def __setstate__(self, word_re):
		self.word_re = word_re

	def __call__(self, text):
		tokens = re.findall(self.word_re, text)
		# Remove empty tokens
		tokens = [token for token in tokens if token]
		return tuple(tokens)

class Splitter:

	def __init__(self, word_separator_re = "\s+"):
		self.word_separator_re = word_separator_re

	def __getstate__(self):
		return self.word_separator_re

	def __setstate__(self, word_separator_re):
		self.word_separator_re = word_separator_re

	def __call__(self, text):
		tokens = re.split(self.word_separator_re, text)
		# Trim tokens by removing leading and trailing puncutation characters
		tokens = [token.strip(punctuation) for token in tokens]
		# Remove empty tokens
		tokens = [token for token in tokens if token]
		return tuple(tokens)
