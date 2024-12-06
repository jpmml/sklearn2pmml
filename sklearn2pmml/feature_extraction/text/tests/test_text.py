from io import BytesIO
from sklearn2pmml.feature_extraction.text import Matcher, Splitter
from unittest import TestCase

import joblib

def _clone(x):
	out_buf = BytesIO()
	joblib.dump(x, out_buf)
	out_buf.flush()
	in_buf = BytesIO(out_buf.getvalue())
	return joblib.load(in_buf)

class MatcherTest(TestCase):

	def test_call(self):
		matcher = Matcher()
		self.assertEqual((), matcher(""))
		self.assertEqual((), matcher("."))
		self.assertEqual(("one", ), matcher("one"))
		self.assertEqual(("one", ), matcher("++one"))
		self.assertEqual(("one", ), matcher("one++"))
		self.assertEqual(("one", "two", "three"), matcher("one two three"))
		self.assertEqual(("one", "_t", "w", "o_", "three"), matcher(",one _t,w.o_ three."))
		matcher = Matcher("\w{2,}")
		self.assertEqual(("one", "two", "three"), matcher("one two three"))
		self.assertEqual(("one", "_t", "o_", "three"), matcher(",one _t,w.o_ three."))
		matcher = Matcher("\w{4,}")
		self.assertEqual(("three", ), matcher("one two three"))
		self.assertEqual(("three", ), matcher(",one _t,w.o_ three."))

	def test_pickle(self):
		matcher = Matcher("\S+")
		self.assertEqual("\S+", matcher.word_re)
		matcher_clone = _clone(matcher)
		self.assertEqual("\S+", matcher_clone.word_re)

class SplitterTest(TestCase):

	def test_call(self):
		splitter = Splitter()
		self.assertEqual((), splitter(""))
		self.assertEqual((), splitter("."))
		self.assertEqual(("one", ), splitter("one"))
		self.assertEqual(("++one", ), splitter("++one"))
		self.assertEqual(("one++", ), splitter("one++"))
		self.assertEqual(("one", ), splitter("--one"))
		self.assertEqual(("one", ), splitter("one--"))
		self.assertEqual(("one", "two", "three"), splitter("one two three"))
		self.assertEqual(("one", "t,w.o", "three"), splitter(",one _t,w.o_ three."))

	def test_pickle(self):
		splitter = Splitter("\W")
		self.assertEqual("\W", splitter.word_separator_re)
		splitter_clone = _clone(splitter)
		self.assertEqual("\W", splitter_clone.word_separator_re)
