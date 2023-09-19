from pandas import DataFrame
from sklearn2pmml.cross_reference import make_memorizer_union, make_recaller_union, Memorizer, Recaller
from unittest import TestCase

import numpy

class MemorizerTest(TestCase):

	def test_fit_transform(self):
		memory = dict()
		self.assertEqual(0, len(memory))
		memorizer = Memorizer(memory, ["int"])
		X = numpy.asarray([[-1], [1]])
		Xt = memorizer.fit_transform(X)
		self.assertEqual((2, 0), Xt.shape)
		self.assertEqual(1, len(memory))
		self.assertEqual([-1, 1], memory["int"].tolist())

		memory = DataFrame()
		self.assertEqual((0, 0), memory.shape)
		memorizer = Memorizer(memory, ["int", "float", "str"])
		X = numpy.asarray([[1, 1.0, "one"], [2, 2.0, "two"], [3, 3.0, "three"]])
		Xt = memorizer.fit_transform(X)
		self.assertEqual((3, 0), Xt.shape)
		self.assertEqual((3, 3), memory.shape)
		self.assertEqual(["1", "2", "3"], memory["int"].tolist())
		self.assertEqual([1, 2, 3], memory["int"].astype(int).tolist())
		self.assertEqual([str(1.0), str(2.0), str(3.0)], memory["float"].tolist())
		self.assertEqual([1.0, 2.0, 3.0], memory["float"].astype(float).tolist())
		self.assertEqual(["one", "two", "three"], memory["str"].tolist())

class RecallerTest(TestCase):

	def test_fit_transform(self):
		memory = {
			"int": [-1, 1]
		}
		recaller = Recaller(memory, ["int"])
		X = numpy.empty((2, 1), dtype = str)
		Xt = recaller.fit_transform(X)
		self.assertEqual((2, 1), Xt.shape)
		self.assertEqual([-1, 1], Xt[:, 0].tolist())

		memory = DataFrame([[1, 1.0, "one"], [2, 2.0, "two"], [3, 3.0, "three"]], columns = ["int", "float", "str"])
		self.assertEqual((3, 3), memory.shape)
		recaller = Recaller(memory, ["int"])
		X = numpy.empty((3, 5), dtype = str)
		Xt = recaller.fit_transform(X)
		self.assertEqual((3, 1), Xt.shape)
		self.assertEqual([1, 2, 3], Xt[:, 0].tolist())
		recaller = Recaller(memory, ["int", "float", "str"])
		Xt = recaller.fit_transform(X)
		self.assertEqual((3, 3), Xt.shape)
		self.assertEqual([1, 2, 3], Xt[:, 0].tolist())
		self.assertEqual([1.0, 2.0, 3.0], Xt[:, 1].tolist())
		self.assertEqual(["one", "two", "three"], Xt[:, 2].tolist())

class FunctionTest(TestCase):

	def test_make_memorizer_union(self):
		memory = dict()
		self.assertEqual(0, len(memory))
		memorizer = make_memorizer_union(memory, ["int"])
		X = numpy.asarray([[-1], [1]])
		Xt = memorizer.fit_transform(X)
		self.assertEqual((2, 1), Xt.shape)
		self.assertEqual(1, len(memory))
		self.assertEqual([-1, 1], memory["int"].tolist())

	def test_make_recaller_union(self):
		memory = {
			"int": [-1, 1]
		}
		recaller = make_recaller_union(memory, ["int"])
		X = numpy.full((2, 1), 0, dtype = int)
		Xt = recaller.fit_transform(X)
		self.assertEqual((2, 2), Xt.shape)
		self.assertEqual([-1, 1], Xt[:, 0].tolist())
		self.assertEqual([0, 0], Xt[:, 1].tolist())
