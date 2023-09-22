from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn2pmml.preprocessing import IdentityTransformer

import numpy

class Memory(object):

	def __init__(self):
		self.data = dict()

	def __getitem__(self, key):
		return self.data[key]

	def __setitem__(self, key, value):
		self.data[key] = value

	def clear(self):
		self.data.clear()

	def __len__(self):
		return len(self.data)

	def __getstate__(self):
		state = self.__dict__.copy()
		state["data"] = dict()
		return state

	def __setstate__(self, state):
		self.__dict__.update(state)

def make_memorizer_union(memory, names):
	return FeatureUnion([
		("memorizer", Memorizer(memory, names)),
		("identity", IdentityTransformer()),
	])

def make_recaller_union(memory, names):
	return FeatureUnion([
		("recaller", Recaller(memory, names)),
		("identity", IdentityTransformer())
	])

class _BaseMemoryManager(BaseEstimator, TransformerMixin):

	def __init__(self, memory, names):
		self.memory = memory
		if not isinstance(names, list):
			raise TypeError()
		self.names = names

class Memorizer(_BaseMemoryManager):

	def __init__(self, memory, names):
		super(Memorizer, self).__init__(memory, names)

	def fit(self, X, y = None):
		if X.shape[1] != len(self.names):
			raise ValueError()
		return self

	def transform(self, X):
		for idx, name in enumerate(self.names):
			if isinstance(X, DataFrame):
				x = X.iloc[:, idx]
			else:
				x = X[:, idx]
			self.memory[name] = x.copy()
		return numpy.empty(shape = (X.shape[0], 0), dtype = int)

class Recaller(_BaseMemoryManager):

	def __init__(self, memory, names):
		super(Recaller, self).__init__(memory, names)

	def fit(self, X, y = None):
		return self

	def transform(self, X):
		result = []
		for idx, name in enumerate(self.names):
			x = self.memory[name]
			result.append(x.copy())
		return numpy.asarray(result).T
