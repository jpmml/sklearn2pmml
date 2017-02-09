from io import BytesIO
from sklearn.externals import joblib
from sklearn2pmml.feature_extraction.text import Splitter
from unittest import TestCase

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
		self.assertEqual("\W", splitter.separator_re)
		splitter_clone = SplitterTest._clone(splitter)
		self.assertEqual("\W", splitter_clone.separator_re)

	@staticmethod
	def _clone(x):
		out_buf = BytesIO()
		joblib.dump(x, out_buf)
		out_buf.flush()
		in_buf = BytesIO(out_buf.getvalue())
		return joblib.load(in_buf)
