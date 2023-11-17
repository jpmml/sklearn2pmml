from pandas import DataFrame
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn2pmml.cross_reference import make_memorizer_union, make_recaller_union, Memory, Memorizer, Recaller
from sklearn2pmml.preprocessing import IdentityTransformer
from unittest import TestCase

import copy
import numpy
import pickle

class MemoryTest(TestCase):

	def _workflow(self, data):
		memory = Memory(data)
		self.assertIs(data, memory.data)
		self.assertEqual(0, len(memory))
		memory["int"] = [1, 2, 3]
		self.assertEqual(1, len(memory))
		self.assertTrue("int" in memory)
		self.assertEqual([1, 2, 3], numpy.asarray(memory["int"]).tolist())
		memory_clone = pickle.loads(pickle.dumps(memory))
		self.assertIsNot(memory, memory_clone)
		self.assertIsNot(memory.data, memory_clone.data)
		self.assertEqual(0, len(memory_clone))
		self.assertFalse("float" in memory)
		with self.assertRaises(KeyError):
			memory["float"]
		memory["float"] = [1.0, 2.0, 3.0]
		self.assertEqual(2, len(memory))
		self.assertTrue("float" in memory)
		self.assertEqual([1.0, 2.0, 3.0], numpy.asarray(memory["float"]).tolist())
		del memory["float"]
		self.assertEqual(1, len(memory))
		memory.clear()
		self.assertEqual(0, len(memory))
		self.assertIs(data, memory.data)

	def test_dict_workflow(self):
		data = dict()
		self._workflow(data)

	def test_dataframe_workflow(self):
		data = DataFrame()
		self._workflow(data)

	def test_copy(self):
		memory = Memory()
		memory_copy = copy.copy(memory)
		self.assertIs(memory, memory_copy)
		memory_deepcopy = copy.deepcopy(memory)
		self.assertIs(memory, memory_deepcopy)

class MemorizerTest(TestCase):

	def test_transform(self):
		memory = dict()
		self.assertEqual(0, len(memory))
		memorizer = Memorizer(memory, ["int"])
		X = numpy.asarray([[-1], [1]])
		memorizer.fit(X)
		self.assertEquals(0, len(memory))
		Xt = memorizer.transform(X)
		self.assertEqual((2, 0), Xt.shape)
		self.assertEqual(1, len(memory))
		self.assertEqual([-1, 1], memory["int"].tolist())

		memory = dict()
		self.assertEquals(0, len(memory))
		memorizer = Memorizer(memory, ["int"], transform_only = False)
		memorizer.fit(X)
		self.assertEquals(1, len(memory))

		memory = DataFrame()
		self.assertEqual((0, 0), memory.shape)
		memorizer = Memorizer(memory, ["int", "float", "str"])
		X = numpy.asarray([[1, 1.0, "one"], [2, 2.0, "two"], [3, 3.0, "three"]])
		pipeline = make_pipeline(memorizer)
		Xt = pipeline.fit_transform(X)
		self.assertEqual((3, 0), Xt.shape)
		self.assertEqual((3, 3), memory.shape)
		self.assertEqual(["1", "2", "3"], memory["int"].tolist())
		self.assertEqual([1, 2, 3], memory["int"].astype(int).tolist())
		self.assertEqual([str(1.0), str(2.0), str(3.0)], memory["float"].tolist())
		self.assertEqual([1.0, 2.0, 3.0], memory["float"].astype(float).tolist())
		self.assertEqual(["one", "two", "three"], memory["str"].tolist())

	def test_clone(self):
		memory = Memory()
		memorizer = Memorizer(memory, ["flag"])
		memorizer_clone = clone(memorizer)
		self.assertIsNot(memorizer, memorizer_clone)
		self.assertIs(memorizer.memory, memorizer_clone.memory)

class RecallerTest(TestCase):

	def test_transform(self):
		memory = {
			"int": [-1, 1]
		}
		recaller = Recaller(memory, ["int"])
		X = numpy.empty((2, 1), dtype = str)
		recaller.fit(X)
		self.assertEqual(1, len(memory))
		Xt = recaller.transform(X)
		self.assertEqual((2, 1), Xt.shape)
		self.assertEqual([-1, 1], Xt[:, 0].tolist())

		memory = DataFrame([[1, 1.0, "one"], [2, 2.0, "two"], [3, 3.0, "three"]], columns = ["int", "float", "str"])
		self.assertEqual((3, 3), memory.shape)
		recaller = Recaller(memory, ["int"], clear_after = False)
		pipeline = make_pipeline(recaller)
		X = numpy.empty((3, 5), dtype = str)
		Xt = pipeline.fit_transform(X)
		self.assertEqual((3, 3), memory.shape)
		self.assertEqual((3, 1), Xt.shape)
		self.assertEqual([1, 2, 3], Xt[:, 0].tolist())
		recaller = Recaller(memory, ["int", "float", "str"], clear_after = True)
		pipeline = make_pipeline(recaller)
		Xt = pipeline.fit_transform(X)
		self.assertEqual((3, 0), memory.shape)
		self.assertEqual((3, 3), Xt.shape)
		self.assertEqual([1, 2, 3], Xt[:, 0].tolist())
		self.assertEqual([1.0, 2.0, 3.0], Xt[:, 1].tolist())
		self.assertEqual(["one", "two", "three"], Xt[:, 2].tolist())
		with self.assertRaises(KeyError):
			recaller.transform(X)

	def test_clone(self):
		memory = Memory()
		recaller = Recaller(memory, ["flag"])
		recaller_clone = clone(recaller)
		self.assertIsNot(recaller, recaller_clone)
		self.assertIs(recaller.memory, recaller_clone.memory)

class FunctionTest(TestCase):

	def test_make_memorizer_union(self):
		memory = dict()
		self.assertEqual(0, len(memory))
		memorizer_union = make_memorizer_union(memory, ["int"], position = "first")
		self.assertIsInstance(memorizer_union.transformer_list[0][1], Memorizer)
		self.assertIsInstance(memorizer_union.transformer_list[1][1], IdentityTransformer)
		memorizer_union = make_memorizer_union(memory, ["int"], position = "last")
		self.assertIsInstance(memorizer_union.transformer_list[0][1], IdentityTransformer)
		self.assertIsInstance(memorizer_union.transformer_list[1][1], Memorizer)
		X = numpy.asarray([[-1], [1]])
		Xt = memorizer_union.fit_transform(X)
		self.assertEqual((2, 1), Xt.shape)
		self.assertEqual(1, len(memory))
		self.assertEqual([-1, 1], memory["int"].tolist())

	def test_make_recaller_union(self):
		memory = {
			"int": [-1, 1]
		}
		recaller_union = make_recaller_union(memory, ["int"])
		X = numpy.full((2, 1), 0, dtype = int)
		Xt = recaller_union.fit_transform(X)
		self.assertEqual((2, 2), Xt.shape)
		self.assertEqual([-1, 1], Xt[:, 0].tolist())
		self.assertEqual([0, 0], Xt[:, 1].tolist())
