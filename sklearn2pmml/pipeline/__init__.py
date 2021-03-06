from pandas import DataFrame, Series
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.pipeline import Pipeline

import numpy

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

def _filter_column_names(X):
	return (numpy.asarray(X)).astype(str);

def _get_column_names(X):
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
	if isinstance(X, DataFrame):
		return X.values
	elif isinstance(X, Series):
		return X.values
	# elif isinstance(X, H2OFrame)
	elif hasattr(X, "as_data_frame"):
		X = X.as_data_frame()
		return X.values
	else:
		return X

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

	def predict_proba(self, X, **predict_proba_params):
		Xt = X
		if hasattr(self, "_iter"):
			for _, name, transform in self._iter(with_final = False):
				Xt = transform.transform(Xt)
		else:
			for name, transform in self.steps[:-1]:
				if transform is not None:
					Xt = transform.transform(Xt)
		return self.steps[-1][-1].predict_proba(Xt, **predict_proba_params)

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

	def configure(self, **pmml_options):
		if len(pmml_options) > 0:
			estimator = self._final_estimator
			while isinstance(estimator, Pipeline):
				estimator = estimator._final_estimator
			estimator.pmml_options_ = dict()
			estimator.pmml_options_.update(pmml_options)

	def verify(self, X, predict_params = {}, predict_proba_params = {}, precision = 1e-13, zeroThreshold = 1e-13):
		active_fields = _get_column_names(X)
		if self.active_fields is None or active_fields is None:
			raise ValueError("Cannot perform model validation with anonymous data")
		if self.active_fields.tolist() != active_fields.tolist():
			raise ValueError("The columns between training data {} and verification data {} do not match".format(self.active_fields, active_fields))
		active_values = _get_values(X)
		y = self.predict(X, **predict_params)
		target_values = _get_values(y)
		estimator = self._final_estimator
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
