from sklearn2pmml.util import sizeof, deep_sizeof
from unittest import TestCase

import numpy

class MeasurementTest(TestCase):

	def test_sizeof(self):
		self.assertEqual(sizeof(None, with_overhead = False), sizeof(None, with_overhead = True))

		self.assertEqual(sizeof(1, with_overhead = False), sizeof(1, with_overhead = True))
		self.assertEqual(sizeof(True, with_overhead = False), sizeof(True, with_overhead = True))

		self.assertEqual(sizeof(1), sizeof(2))
		self.assertEqual(sizeof(1.0), sizeof(2.0))

		self.assertTrue(sizeof([], with_overhead = False) < sizeof([], with_overhead = True))

		self.assertTrue(sizeof(["a"]) > sizeof([]))
		self.assertTrue(sizeof(["abc"], with_overhead = False) == sizeof(["a"], with_overhead = False))
		self.assertTrue(sizeof(["abc"], with_overhead = True) == sizeof(["a"], with_overhead = True))
		self.assertTrue(sizeof(["aaa", "bbb", "ccc"], with_overhead = False) == sizeof(["a", "b", "c"], with_overhead = False))
		self.assertTrue(sizeof(["aaa", "bbb", "ccc"], with_overhead = True) == sizeof(["a", "b", "c"], with_overhead = True))

		self.assertTrue(sizeof((), with_overhead = False) < sizeof((), with_overhead = True))

		self.assertTrue(sizeof(("a")) > sizeof(()))
		self.assertTrue(sizeof(("abc")) == sizeof(("a")) + 2)
		self.assertTrue(sizeof(("aaa", "bbb", "ccc"), with_overhead = False) == sizeof(("a", "b", "c"), with_overhead = False))
		self.assertTrue(sizeof(("aaa", "bbb", "ccc"), with_overhead = True) == sizeof(("a", "b", "c"), with_overhead = True))

	def test_deep_sizeof(self):
		self.assertTrue(deep_sizeof([], with_overhead = False) < deep_sizeof([], with_overhead = True))

		self.assertEqual(deep_sizeof(["abc"], with_overhead = False), deep_sizeof(["a"], with_overhead = False) + 2)
		self.assertEqual(deep_sizeof(["abc"], with_overhead = True), deep_sizeof(["a"], with_overhead = True) + 2)
		self.assertEqual(deep_sizeof(["aaa", "bbb", "ccc"], with_overhead = False), deep_sizeof(["a", "b", "c"], with_overhead = False) + 6)
		self.assertEqual(deep_sizeof(["aaa", "bbb", "ccc"], with_overhead = True), deep_sizeof(["a", "b", "c"], with_overhead = True) + 6)

		self.assertTrue(deep_sizeof((), with_overhead = False) < deep_sizeof((), with_overhead = True))

		self.assertEqual(deep_sizeof(("aaa", "bbb", "ccc"), with_overhead = False), deep_sizeof(("a", "b", "c"), with_overhead = False) + 6)
		self.assertEqual(deep_sizeof(("aaa", "bbb", "ccc"), with_overhead = True), deep_sizeof(("a", "b", "c"), with_overhead = True) + 6)
