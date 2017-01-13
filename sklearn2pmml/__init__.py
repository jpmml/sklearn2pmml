#!/usr/bin/env python

from pandas import DataFrame, Series
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline

import os
import pandas
import pkg_resources
import platform
import sklearn
import sklearn_pandas
import subprocess
import tempfile

__copyright__ = "Copyright (c) 2015 Villu Ruusmann"
__license__ = "GNU Affero General Public License (AGPL) version 3.0"
__version__ = "0.15.1"

class PMMLPipeline(Pipeline):

	def __init__(self, steps):
		Pipeline.__init__(self, steps)

	def _fit(self, X, y, **fit_params):
		if(isinstance(X, DataFrame)):
			self.active_fields = X.columns.values
		if(isinstance(y, Series)):
			self.target_field = y.name
		return Pipeline._fit(self, X, y, **fit_params)

def _package_classpath():
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

def sklearn2pmml(pipeline, pmml, user_classpath = [], with_repr = False, debug = False):
	"""Converts fitted Scikit-Learn pipeline to PMML.

	Parameters:
	----------
	pipeline: PMMLPipeline
		The pipeline.

	pmml: string
		The path to where the PMML document should be stored.

	user_classpath: list of strings, optional
		The paths to JAR files that provide custom Transformer, Selector and/or Estimator converter classes.
		The JPMML-SkLearn classpath is constructed by appending user JAR files to package JAR files.

	with_repr: boolean, optional
		If true, insert the textual representation of pipeline into the PMML document.

	debug: boolean, optional
		If true, print information about the conversion operation.

	"""
	if(debug):
		print("python: ", platform.python_version())
		print("sklearn: ", sklearn.__version__)
		print("sklearn.externals.joblib:", joblib.__version__)
		print("pandas: ", pandas.__version__)
		print("sklearn_pandas: ", sklearn_pandas.__version__)
		print("sklearn2pmml: ", __version__)
	if(not isinstance(pipeline, PMMLPipeline)):
		raise TypeError("The pipeline object is not an instance of " + PMMLPipeline.__name__)
	cmd = ["java", "-cp", os.pathsep.join(_package_classpath() + user_classpath), "org.jpmml.sklearn.Main"]
	dumps = []
	try:
		pipeline_pkl = _dump(pipeline, "pipeline")
		cmd.extend(["--pkl-pipeline-input", pipeline_pkl])
		dumps.append(pipeline_pkl)
		if(with_repr):
			pipeline_repr = repr(pipeline)
			cmd.extend(["--repr-pipeline", pipeline_repr])
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
