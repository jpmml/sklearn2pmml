from sklearn2pmml.preprocessing.regex import REEngine
from unittest import TestCase

class REEngineTest(TestCase):

	def test_matches(self):
		engine = REEngine(r"ar?y")
		self.assertTrue(engine.matches("January"))
		self.assertTrue(engine.matches("February"))
		self.assertFalse(engine.matches("March"))
		self.assertFalse(engine.matches("April"))
		self.assertTrue(engine.matches("May"))
		self.assertFalse(engine.matches("June"))

	def test_replace(self):
		engine = REEngine(r"(\w)")
		self.assertEqual("P u p p y", engine.replace(r"\1 ", "Puppy").strip())
