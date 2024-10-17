from sklearn2pmml.preprocessing.regex import PCRE2Engine, REEngine
from unittest import TestCase

class PCRE2EngineTest(TestCase):

	def test_matches(self):
		engine = PCRE2Engine(r"ar?y")
		self.assertTrue(engine.matches("January"))
		self.assertFalse(engine.matches("March"))
		self.assertTrue(engine.matches("May"))
		engine = PCRE2Engine(r"r$")
		self.assertFalse(engine.matches("March"))
		self.assertTrue(engine.matches("October"))

	def test_replace(self):
		engine = PCRE2Engine(r"(\w)")
		self.assertEqual("P u p p y", engine.replace(r"$1 ", "Puppy").rstrip())
		self.assertEqual(r"\1 \1 \1 \1 \1", engine.replace(r"\1 ", "Puppy").rstrip())
		with self.assertRaises(Exception):
			engine.replace(r"$", "Puppy")
		self.assertEqual("$$$$$", engine.replace(r"$$", "Puppy"))

class REEngineTest(TestCase):

	def test_matches(self):
		engine = REEngine(r"ar?y")
		self.assertTrue(engine.matches("January"))
		self.assertTrue(engine.matches("February"))
		self.assertFalse(engine.matches("March"))
		self.assertFalse(engine.matches("April"))
		self.assertTrue(engine.matches("May"))
		self.assertFalse(engine.matches("June"))
		engine = REEngine(r"r$")
		self.assertFalse(engine.matches("March"))
		self.assertTrue(engine.matches("October"))

	def test_replace(self):
		engine = REEngine(r"(\w)")
		self.assertEqual("$1 $1 $1 $1 $1", engine.replace(r"$1 ", "Puppy").rstrip())
		self.assertEqual("P u p p y", engine.replace(r"\1 ", "Puppy").rstrip())
		self.assertEqual("$$$$$", engine.replace(r"$", "Puppy"))
