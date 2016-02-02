#!/usr/bin/env python

from sklearn.base import BaseEstimator
from sklearn_pandas import DataFrameMapper

import joblib
import os
import pkg_resources
import subprocess
import tempfile

__copyright__ = "Copyright (c) 2015 Villu Ruusmann"
__license__ = "GNU Affero General Public License (AGPL) version 3.0"

def _classpath():
	jars = []
	resources = pkg_resources.resource_listdir("sklearn2pmml.resources", "")
	for resource in resources:
		if(resource.endswith(".jar")):
			jars.append(pkg_resources.resource_filename("sklearn2pmml.resources", resource))
	return jars

def _dump(obj):
	fd, path = tempfile.mkstemp(suffix = ".pkl")
	joblib.dump(obj, path, compress = 9)
	return path

def sklearn2pmml(estimator, mapper, pmml, with_repr = False, verbose = False):
	if(not isinstance(estimator, BaseEstimator)):
		raise TypeError("The estimator object is not an instance of " + BaseEstimator.__name__)
	if((mapper is not None) and (not isinstance(mapper, DataFrameMapper))):
		raise TypeError("The mapper object is not an instance of " + DataFrameMapper.__name__)
	cmd = ["java", "-cp", os.pathsep.join(_classpath()), "org.jpmml.sklearn.Main"]
	dumps = []
	try:
		estimator_pkl = _dump(estimator)
		cmd.extend(["--pkl-estimator-input", estimator_pkl])
		dumps.append(estimator_pkl)
		if(with_repr):
			estimator_repr = repr(estimator)
			cmd.extend(["--repr-estimator", estimator_repr])
		if(mapper):
			mapper_pkl = _dump(mapper)
			cmd.extend(["--pkl-mapper-input", mapper_pkl])
			dumps.append(mapper_pkl)
			if(with_repr):
				mapper_repr = repr(mapper)
				cmd.extend(["--repr-mapper", mapper_repr])
		cmd.extend(["--pmml-output", pmml])
		if(verbose):
			print(cmd)
		subprocess.check_call(cmd)
	finally:
		for dump in dumps:
			os.remove(dump)