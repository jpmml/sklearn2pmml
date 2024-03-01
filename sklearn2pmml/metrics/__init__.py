from sklearn.metrics import accuracy_score, confusion_matrix, fbeta_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score
from sklearn2pmml.util.pmml import make_element
from sklearn2pmml.util.pmml import Array, PMMLElement

import numpy

class ConfusionMatrix(PMMLElement):

	def __init__(self, labels, confusion_matrix):
		super(ConfusionMatrix, self).__init__(make_element("ConfusionMatrix"))

		pmml_class_labels = make_element("ClassLabels")
		pmml_class_labels.append(Array(labels).root)
		self.append(pmml_class_labels)

		pmml_matrix = make_element("Matrix")
		for row in confusion_matrix:
			pmml_matrix.append(Array(row).root)
		self.append(pmml_matrix)

class ModelExplanation(PMMLElement):

	def __init__(self):
		super(ModelExplanation, self).__init__(make_element("ModelExplanation"))

class PredictiveModelQuality(PMMLElement):

	def __init__(self, estimator, X, y, target_field, data_usage, data_name):
		super(PredictiveModelQuality, self).__init__(make_element("PredictiveModelQuality", targetField = target_field))

		self.estimator = estimator

		self.X = X
		self.y = y

		# XXX
		self.yt = estimator.predict(X)

		if data_usage is not None:
			data_usages = ["training", "test", "validation"]
			if data_usage not in data_usages:
				raise ValueError("Data usage {0} not in {1}".format(data_usage, data_usages))
			self.set("dataUsage", data_usage)
		if data_name is not None:
			self.set("dataName", data_name)

		n_samples = X.shape[0]
		n_features = (estimator.n_features_in_ if hasattr(estimator, "n_features_in_") else X.shape[1])

		self.set("numOfRecords", n_samples)
		self.set("numOfPredictors", n_features)

class ClassifierQuality(PredictiveModelQuality):

	def __init__(self, estimator, X, y, target_field, data_usage = "training", data_name = None):
		super(ClassifierQuality, self).__init__(estimator = estimator, X = X, y = y, target_field = target_field, data_usage = data_usage, data_name = data_name)

	@property
	def classes_(self):
		return self.estimator.classes_

	def with_all_metrics(self):
		return self \
			.with_accuracy() \
			.with_confusion_matrix()

	def with_accuracy(self):
		accuracy = accuracy_score(self.y, self.yt)
		return self.set("accuracy", accuracy)

	def with_confusion_matrix(self):
		pmml_confusion_matrix = ConfusionMatrix(labels = self.classes_, confusion_matrix = confusion_matrix(self.y, self.yt, normalize = None))
		return self.append(pmml_confusion_matrix)

class BinaryClassifierQuality(ClassifierQuality):

	def __init__(self, estimator, X, y, target_field, data_usage = "training", data_name = None):
		super(BinaryClassifierQuality, self).__init__(estimator = estimator, X = X, y = y, target_field = target_field, data_usage = data_usage, data_name = data_name)

		if len(self.classes_) != 2:
			raise ValueError()

	def with_all_metrics(self):
		return super(BinaryClassifierQuality, self).with_all_metrics() \
			.with_precision() \
			.with_recall() \
			.with_F1() \
			.with_F2() \
			.with_Fhalf()

	def with_precision(self):
		classes = self.classes_
		precision = precision_score(self.y, self.yt, labels = classes, pos_label = classes[1])
		return self.set("precision", precision)

	def with_recall(self):
		classes = self.classes_
		recall = recall_score(self.y, self.yt, labels = classes, pos_label = classes[1])
		return self.set("recall", recall)

	def with_F1(self):
		return self.set("F1", self._fbeta_score(1.0))

	def with_F2(self):
		return self.set("F2", self._fbeta_score(2.0))

	def with_Fhalf(self):
		return self.set("Fhalf", self._fbeta_score(0.5))

	def _fbeta_score(self, beta):
		classes = self.classes_
		return fbeta_score(self.y, self.yt, beta = beta, labels = classes, pos_label = classes[1])

class RegressorQuality(PredictiveModelQuality):
	
	def __init__(self, estimator, X, y, target_field, data_usage = "training", data_name = None):
		super(RegressorQuality, self).__init__(estimator = estimator, X = X, y = y, target_field = target_field, data_usage = data_usage, data_name = data_name)

	def with_all_metrics(self):
		return self \
			.with_ME() \
			.with_MAE() \
			.with_MSE() \
			.with_RMSE() \
			.with_R2() \
			.with_SSE() \
			.with_SSR()

	def with_ME(self):
		me = numpy.sum(self.yt - self.y) / self.X.shape[0]
		return self.set("meanError", me)

	def with_MAE(self):
		mae = mean_absolute_error(self.y, self.yt)
		return self.set("meanAbsoluteError", mae)

	def with_MSE(self):
		mse = mean_squared_error(self.y, self.yt)
		return self.set("meanSquaredError", mse)

	def with_RMSE(self):
		rmse = numpy.sqrt(mean_squared_error(self.y, self.yt))
		return self.set("rootMeanSquaredError", rmse)

	def with_R2(self):
		r2 = r2_score(self.y, self.yt)
		return self.set("r-squared", r2)

	def with_SSE(self):
		sse = numpy.sum((self.y - self.yt) ** 2)
		return self.set("sumSquaredError", sse)

	def with_SSR(self):
		ymean = numpy.mean(self.y)
		ssr = numpy.sum((ymean - self.yt) ** 2)
		return self.set("sumSquaredRegression", ssr)
