from pandas import DataFrame, Series
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn2pmml.configuration import add_options, clear_options
from sklearn2pmml.customization import add_customizations, clear_customizations, Customization
from sklearn2pmml.util import to_numpy

import numpy
import warnings

class _Verification(object):

	def __init__(self, active_values, target_values, precision, zeroThreshold):
		self.active_values = active_values
		self.target_values = target_values
		if precision < 0:
			raise ValueError("Precision cannot be negative")
		if zeroThreshold < 0:
			raise ValueError("Zero threshold cannot be negative")
		self.precision = precision
		self.zeroThreshold = zeroThreshold

def _get_column_names(X):
	def _filter_column_names(X):
		return (numpy.asarray(X)).astype(str)

	if isinstance(X, DataFrame):
		return _filter_column_names(X.columns.values)
	elif isinstance(X, Series):
		return _filter_column_names(X.name)
	# elif isinstance(X, H2OFrame)
	elif hasattr(X, "names"):
		return _filter_column_names(X.names)
	else:
		return None

def _get_values(X):
	# if isinstance(X, H2OFrame)
	if hasattr(X, "as_data_frame"):
		X = X.as_data_frame()
	return to_numpy(X)

class PMMLPipeline(Pipeline):

	def __init__(self, steps, header = None, predict_transformer = None, predict_proba_transformer = None, apply_transformer = None, memory = None, verbose = False):
		# SkLearn 0.23
		#super(PMMLPipeline, self).__init__(steps = steps, memory = memory, verbose = verbose)
		self.header = header
		self.predict_transformer = predict_transformer
		self.predict_proba_transformer = predict_proba_transformer
		self.apply_transformer = apply_transformer
		# SkLearn 0.24+
		super(PMMLPipeline, self).__init__(steps = steps, memory = memory, verbose = verbose)

	def __repr__(self):
		class_name = self.__class__.__name__
		return "%s(steps=[%s])" % (class_name, (",\n" + (1 + len(class_name) // 2) * " ").join(repr(step) for step in self.steps))

	def _collect_metadata(self, X, y = None):
		# Feature name(s)
		active_fields = _get_column_names(X)
		if active_fields is not None:
			self.active_fields = active_fields
		else:
			warnings.warn("X is missing feature names. The reproducibility of predictions between Scikit-Learn and PMML can not be guaranteed")
		# Label name(s)
		target_fields = _get_column_names(y)
		if target_fields is not None:
			self.target_fields = target_fields
		else:
			if y is not None:
				warnings.warn("y is missing target field name(s)")

	def fit(self, X, y = None, **fit_params):
		self._collect_metadata(X = X, y = y)
		return super(PMMLPipeline, self).fit(X = X, y = y, **fit_params)

	def fit_transform(self, X, y = None, **fit_params):
		self._collect_metadata(X = X, y = y)
		return super(PMMLPipeline, self).fit_transform(X = X, y = y, **fit_params)

	def _transform(self, X):
		Xt = X
		if hasattr(self, "_iter"):
			for _, name, transform in self._iter(with_final = False):
				Xt = transform.transform(Xt)
		else:
			for name, transform in self.steps[:-1]:
				if transform is not None:
					Xt = transform.transform(Xt)
		return Xt

	def apply(self, X):
		Xt = self._transform(X)
		yt = self.steps[-1][-1].apply(Xt)
		if len(yt.shape) == 1:
			yt = yt.reshape(-1, 1)
		return yt

	def predict_transform(self, X, **predict_params):
		y_pred = self.predict(X, **predict_params)
		if self.predict_transformer is not None:
			nrow = len(y_pred)
			y_pred = y_pred.reshape(nrow, -1)
			y_predt = self.predict_transformer.transform(y_pred).reshape(nrow, -1)
			return numpy.hstack((y_pred, y_predt))
		return y_pred

	def predict_proba_transform(self, X, **predict_proba_params):
		y_proba = self.predict_proba(X, **predict_proba_params)
		if self.predict_proba_transformer is not None:
			y_probat = self.predict_proba_transformer.transform(y_proba)
			return numpy.hstack((y_proba, y_probat))
		return y_proba

	def apply_transform(self, X):
		y_apply = self.apply(X)
		if self.apply_transformer is not None:
			y_applyt = self.apply_transformer.transform(y_apply)
			return numpy.hstack((y_apply, y_applyt))
		return y_apply

	def _deep_final_estimator(self):
		estimator = self._final_estimator
		while hasattr(estimator, "_final_estimator"):
			estimator = estimator._final_estimator
		return estimator

	def verify(self, X, predict_params = {}, predict_proba_params = {}, precision = 1e-13, zeroThreshold = 1e-13):
		active_fields = _get_column_names(X)
		if self.active_fields is None or active_fields is None:
			raise ValueError("Cannot perform model validation with anonymous data")
		if self.active_fields.tolist() != active_fields.tolist():
			raise ValueError("The columns between training data {0} and verification data {1} do not match".format(self.active_fields, active_fields))
		active_values = _get_values(X)
		y = self.predict(X, **predict_params)
		target_values = _get_values(y)
		estimator = self._deep_final_estimator()
		if isinstance(estimator, BaseEstimator):
			if isinstance(estimator, RegressorMixin):
				self.verification = _Verification(active_values, target_values, precision, zeroThreshold)
			elif isinstance(estimator, ClassifierMixin):
				self.verification = _Verification(active_values, target_values, precision, zeroThreshold)
				if hasattr(estimator, "predict_proba"):
					try:
						y_proba = self.predict_proba(X, **predict_proba_params)
						self.verification.probability_values = _get_values(y_proba)
					except AttributeError:
						pass
		# elif isinstance(estimator, H2OEstimator):
		elif hasattr(estimator, "_estimator_type") and hasattr(estimator, "download_mojo"):
			if estimator._estimator_type == "regressor":
				self.verification = _Verification(active_values, target_values, precision, zeroThreshold)
			elif estimator._estimator_type == "classifier":
				probability_values = target_values[:, 1:]
				target_values = target_values[:, 0]
				self.verification = _Verification(active_values, target_values, precision, zeroThreshold)
				self.verification.probability_values = probability_values

	def configure(self, **options):
		estimator = self._deep_final_estimator()
		if len(options):
			add_options(estimator, **options)
		else:
			clear_options(estimator)

	def customize(self, command = "insert", xpath_expr = None, pmml_element = None):
		estimator = self._deep_final_estimator()
		if command:
			customization = Customization(command = command, xpath_expr = xpath_expr, pmml_element = pmml_element)
			add_customizations(estimator, [customization])
		else:
			clear_customizations(estimator)
