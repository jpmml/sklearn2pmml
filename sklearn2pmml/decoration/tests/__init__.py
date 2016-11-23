from sklearn2pmml.decoration import CategoricalDomain, ContinuousDomain
from unittest import TestCase

import numpy

class CategoricalDomainTest(TestCase):

	def test_fit(self):
		domain = CategoricalDomain()
		self.assertEqual("return_invalid", domain.invalid_value_treatment)
		self.assertFalse(hasattr(domain, "data_"))
		domain = domain.fit(numpy.array([1, 3, 2, 2]))
		self.assertEqual(numpy.array([1, 2, 3]).tolist(), domain.data_.tolist())

class ContinuousDomainTest(TestCase):

	def test_fit(self):
		domain = ContinuousDomain()
		self.assertEqual("return_invalid", domain.invalid_value_treatment)
		self.assertFalse(hasattr(domain, "data_min_"))
		self.assertFalse(hasattr(domain, "data_max_"))
		domain = domain.fit(numpy.array([1.0, 3.0, 2.0, 2.0]))
		self.assertEqual(1.0, domain.data_min_)
		self.assertEqual(3.0, domain.data_max_)
