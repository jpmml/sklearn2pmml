from sklearn2pmml.tpot import make_pmml_config
from unittest import TestCase

import numpy

class FunctionTest(TestCase):

	def test_make_pmml_config(self):
		config = {
			"sklearn.kernel_approximation.RBFSampler" : {"gamma" : numpy.arange(0.0, 1.01, 0.05)},
			"sklearn.preprocessing.StandardScaler" : {}
		}
		pmml_config = make_pmml_config(config)
		self.assertEqual({"sklearn.preprocessing.StandardScaler" : {}}, pmml_config)
