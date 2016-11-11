#!/usr/bin/env python

from sklearn.base import BaseEstimator
from sklearn.externals import joblib
from sklearn_pandas import DataFrameMapper

from metadata import __version__

import os
import pkg_resources
import platform
import sklearn
import sklearn_pandas
import subprocess
import tempfile


def _classpath():
	jars = []
	resources = pkg_resources.resource_listdir("sklearn2pmml.resources", "")
	for resource in resources:
		if(resource.endswith(".jar")):
			jars.append(pkg_resources.resource_filename("sklearn2pmml.resources", resource))
	return jars

def _dump(obj, prefix):
	fd, path = tempfile.mkstemp(prefix = (prefix + "-"), suffix = ".pkl.z")
	try:
		joblib.dump(obj, path, compress = 3)
	finally:
		os.close(fd)
	return path

def sklearn2pmml(estimator, mapper, pmml, with_repr = False, debug = False):
	if(debug):
		print("python: ", platform.python_version())
		print("sklearn: ", sklearn.__version__)
		print("sklearn.externals.joblib:", joblib.__version__)
		print("sklearn_pandas: ", sklearn_pandas.__version__)
		print("sklearn2pmml: ", __version__)
	if(not isinstance(estimator, BaseEstimator)):
		raise TypeError("The estimator object is not an instance of " + BaseEstimator.__name__)
	if((mapper is not None) and (not isinstance(mapper, DataFrameMapper))):
		raise TypeError("The mapper object is not an instance of " + DataFrameMapper.__name__)
	cmd = ["java", "-cp", os.pathsep.join(_classpath()), "org.jpmml.sklearn.Main"]
	dumps = []
	try:
		estimator_pkl = _dump(estimator, "estimator")
		cmd.extend(["--pkl-estimator-input", estimator_pkl])
		dumps.append(estimator_pkl)
		if(with_repr):
			estimator_repr = repr(estimator)
			cmd.extend(["--repr-estimator", estimator_repr])
		if(mapper):
			mapper_pkl = _dump(mapper, "mapper")
			cmd.extend(["--pkl-mapper-input", mapper_pkl])
			dumps.append(mapper_pkl)
			if(with_repr):
				mapper_repr = repr(mapper)
				cmd.extend(["--repr-mapper", mapper_repr])
		cmd.extend(["--pmml-output", pmml])
		if(debug):
			print(" ".join(cmd))
		subprocess.check_call(cmd)
	finally:
		if(debug):
			print("Preserved joblib dump file(s): ", " ".join(dumps))
		else:
			for dump in dumps:
				os.remove(dump)
