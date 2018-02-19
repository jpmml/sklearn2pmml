from pandas import DataFrame, Series
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.externals import joblib
from sklearn.feature_selection.base import SelectorMixin
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn_pandas import DataFrameMapper
from subprocess import PIPE, Popen
from zipfile import ZipFile

import numpy
import os
import pandas
import pkg_resources
import platform
import re
import sklearn
import sklearn_pandas
import tempfile

from .metadata import __copyright__, __license__, __version__

class _Verification(object):

	def __init__(self, active_values, target_values, precision, zeroThreshold):
		if precision < 0:
			raise ValueError("Precision cannot be negative")
		if zeroThreshold < 0:
			raise ValueError("Zero threshold cannot be negative")
		self.active_values = active_values
		self.target_values = target_values
		self.precision = precision
		self.zeroThreshold = zeroThreshold

def _filter_column_names(X):
	return (numpy.asarray(X)).astype(str);

def _get_column_names(X):
	if isinstance(X, DataFrame):
		return _filter_column_names(X.columns.values)
	elif isinstance(X, Series):
		return _filter_column_names(X.name)
	else:
		return None

def _get_values(X):
	if isinstance(X, DataFrame):
		return X.values
	elif isinstance(X, Series):
		return X.values
	else:
		return X

class PMMLPipeline(Pipeline):

	def __init__(self, steps):
		super(PMMLPipeline, self).__init__(steps = steps)

	def __repr__(self):
		class_name = self.__class__.__name__
		return "%s(steps=[%s])" % (class_name, (",\n" + (1 + len(class_name) // 2) * " ").join(repr(step) for step in self.steps))

	def _fit(self, X, y = None, **fit_params):
		# Collect feature name(s)
		active_fields = _get_column_names(X)
		if active_fields is not None:
			self.active_fields = active_fields
		# Collect label name(s)
		target_fields = _get_column_names(y)
		if target_fields is not None:
			self.target_fields = target_fields
		return super(PMMLPipeline, self)._fit(X = X, y = y, **fit_params)

	def verify(self, X, precision = 1e-13, zeroThreshold = 1e-13):
		active_fields = _get_column_names(X)
		if self.active_fields is None or active_fields is None:
			raise ValueError("Cannot perform model validation with anonymous data")
		if self.active_fields.tolist() != active_fields.tolist():
			raise ValueError("The columns between training data {} and verification data {} do not match".format(self.active_fields, active_fields))
		active_values = _get_values(X)
		y = self.predict(X)
		target_values = _get_values(y)
		self.verification = _Verification(active_values, target_values, precision, zeroThreshold)
		estimator = self._final_estimator
		if isinstance(estimator, ClassifierMixin) and hasattr(estimator, "predict_proba"):
			try:
				y_proba = self.predict_proba(X)
				self.verification.probability_values = _get_values(y_proba)
			except AttributeError:
				pass

class EstimatorProxy(BaseEstimator):

	def __init__(self, estimator_, attr_names_ = ["feature_importances_"]):
		self.estimator_ = estimator_
		self.attr_names_ = attr_names_
		try:
			self._copy_attrs()
		except NotFittedError:
			pass

	def __getattr__(self, name):
		return getattr(self.estimator_, name)

	def _copy_attrs(self):
		for attr_name_ in self.attr_names_:
			if hasattr(self.estimator_, attr_name_):
				setattr(self, attr_name_, getattr(self.estimator_, attr_name_))

	def fit(self, X, y = None, **fit_params):
		self.estimator_.fit(X, y, **fit_params)
		self._copy_attrs()
		return self

class SelectorProxy(BaseEstimator):

	def __init__(self, selector_):
		self.selector_ = selector_
		try:
			self._copy_attrs()
		except NotFittedError:
			pass

	def __getattr__(self, name):
		return getattr(self.selector_, name)

	def _copy_attrs(self):
		try:
			setattr(self, "support_mask_", self.selector_._get_support_mask())
		except ValueError:
			pass

	def fit(self, X, y = None, **fit_params):
		self.selector_.fit(X, y, **fit_params)
		self._copy_attrs()
		return self

	def fit_transform(self, X, y = None, **fit_params):
		Xt = self.selector_.fit_transform(X, y, **fit_params)
		self._copy_attrs()
		return Xt

def _get_steps(obj):
	if isinstance(obj, Pipeline):
		return obj.steps
	elif isinstance(obj, BaseEstimator):
		return [("estimator", obj)]
	else:
		raise ValueError()

def _filter(obj):
	if isinstance(obj, DataFrameMapper):
		obj.features = _filter_steps(obj.features)
		if hasattr(obj, "built_features"):
			if obj.built_features is not None:
				obj.built_features = _filter_steps(obj.built_features)
	elif isinstance(obj, FeatureUnion):
		obj.transformer_list = _filter_steps(obj.transformer_list)
	elif isinstance(obj, Pipeline):
		obj.steps = _filter_steps(obj.steps)
	elif isinstance(obj, SelectorMixin):
		if isinstance(obj, SelectFromModel):
			if hasattr(obj, "estimator"):
				setattr(obj, "estimator", EstimatorProxy(obj.estimator))
			if hasattr(obj, "estimator_"):
				setattr(obj, "estimator_", EstimatorProxy(obj.estimator_))
		return SelectorProxy(obj)
	elif isinstance(obj, list):
		return [_filter(e) for e in obj]
	return obj

def _filter_steps(steps):
	return [(step[:1] + (_filter(step[1]), ) + step[2:]) for step in steps]

def make_pmml_pipeline(obj, active_fields = None, target_fields = None):
	"""Translates a regular Scikit-Learn estimator or pipeline to a PMML pipeline.

	Parameters:
	----------
	obj: BaseEstimator
		The object.

	active_fields: list of strings, optional
		Feature names. If missing, "x1", "x2", .., "xn" are assumed.

	target_fields: list of strings, optional
		Label name(s). If missing, "y" is assumed.

	"""
	steps = _filter_steps(_get_steps(obj))
	pipeline = PMMLPipeline(steps)
	if active_fields is not None:
		pipeline.active_fields = numpy.asarray(active_fields)
	if target_fields is not None:
		pipeline.target_fields = numpy.asarray(target_fields)
	return pipeline

def _java_version():
	try:
		process = Popen(["java", "-version"], stdout = PIPE, stderr = PIPE, bufsize = 1)
	except:
		return None
	output, error = process.communicate()
	retcode = process.poll()
	if retcode:
		return None
	match = re.match("^(.*)\sversion\s\"(.*)\"$", error.decode("UTF-8"), re.MULTILINE)
	if match:
		return (match.group(1), match.group(2))
	else:
		return None

def _package_classpath():
	jars = []
	resources = pkg_resources.resource_listdir("sklearn2pmml.resources", "")
	for resource in resources:
		if resource.endswith(".jar"):
			jars.append(pkg_resources.resource_filename("sklearn2pmml.resources", resource))
	return jars

def _classpath(user_classpath):
	return _package_classpath() + user_classpath

def _process_classpath(name, fun, user_classpath):
	jars = _classpath(user_classpath)
	for jar in jars:
		with ZipFile(jar, "r") as zipfile:
			try:
				zipentry = zipfile.getinfo(name)
			except KeyError:
				pass
			else:
				fun(zipfile.open(zipentry))

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
		If true, insert the string representation of pipeline into the PMML document.

	debug: boolean, optional
		If true, print information about the conversion operation.

	"""
	if debug:
		java_version = _java_version()
		if java_version is None:
			java_version = ("java", "N/A")
		print("python: {0}".format(platform.python_version()))
		print("sklearn: {0}".format(sklearn.__version__))
		print("sklearn.externals.joblib: {0}".format(joblib.__version__))
		print("pandas: {0}".format(pandas.__version__))
		print("sklearn_pandas: {0}".format(sklearn_pandas.__version__))
		print("sklearn2pmml: {0}".format(__version__))
		print("{0}: {1}".format(java_version[0], java_version[1]))
	if not isinstance(pipeline, PMMLPipeline):
		raise TypeError("The pipeline object is not an instance of " + PMMLPipeline.__name__)
	cmd = ["java", "-cp", os.pathsep.join(_classpath(user_classpath)), "org.jpmml.sklearn.Main"]
	dumps = []
	try:
		if with_repr:
			pipeline.repr_ = repr(pipeline)
		pipeline_pkl = _dump(pipeline, "pipeline")
		cmd.extend(["--pkl-pipeline-input", pipeline_pkl])
		dumps.append(pipeline_pkl)
		cmd.extend(["--pmml-output", pmml])
		if debug:
			print("Executing command:\n{0}".format(" ".join(cmd)))
		try:
			process = Popen(cmd, stdout = PIPE, stderr = PIPE, bufsize = 1)
		except OSError:
			raise RuntimeError("Java is not installed, or the Java executable is not on system path")
		output, error = process.communicate()
		retcode = process.poll()
		if debug or retcode:
			if(len(output) > 0):
				print("Standard output:\n{0}".format(output.decode("UTF-8")))
			else:
				print("Standard output is empty")
			if(len(error) > 0):
				print("Standard error:\n{0}".format(error.decode("UTF-8")))
			else:
				print("Standard error is empty")
		if retcode:
			raise RuntimeError("The JPMML-SkLearn conversion application has failed. The Java executable should have printed more information about the failure into its standard output and/or standard error streams")
	finally:
		if debug:
			print("Preserved joblib dump file(s): {0}".format(" ".join(dumps)))
		else:
			for dump in dumps:
				os.remove(dump)

def _parse_properties(lines):
	splitter = re.compile("\s*=\s*")
	properties = dict()
	for line in lines:
		line = line.decode("UTF-8").rstrip()
		if line.startswith("#"):
			continue
		key, value = splitter.split(line)
		properties[key] = value
	return properties

def _supported_classes(user_classpath):
	classes = []
	parser = lambda x: classes.extend(_parse_properties(x.readlines()).keys())
	_process_classpath("META-INF/sklearn2pmml.properties", parser, user_classpath)
	return classes

def _strip_module(name):
	parts = name.split(".")
	if len(parts) > 1:
		parts.pop(-2)
		return ".".join(parts)
	return name

def make_tpot_pmml_config(config, user_classpath = []):
	"""Translates a regular TPOT configuration to a PMML-compatible TPOT configuration.

	Parameters:
	----------
	obj: config
		The configuration dictionary.

	user_classpath: list of strings, optional
		The paths to JAR files that provide custom Transformer, Selector and/or Estimator converter classes.
		The JPMML-SkLearn classpath is constructed by appending user JAR files to package JAR files.

	"""
	tpot_keys = set(config.keys())
	classes = _supported_classes(user_classpath)
	pmml_keys = (set(classes)).union(set([_strip_module(class_) for class_ in classes]))
	return { key : config[key] for key in (tpot_keys).intersection(pmml_keys)}
