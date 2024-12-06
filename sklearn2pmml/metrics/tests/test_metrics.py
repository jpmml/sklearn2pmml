from sklearn.datasets import load_diabetes, load_iris
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn2pmml.metrics import BinaryClassifierQuality, ClassifierQuality, ModelExplanation, RegressorQuality
from sklearn2pmml.util.pmml import Extension
from unittest import TestCase

import datetime

def make_model_explanation(estimator, X, y, quality):
	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.75)
	pipeline = Pipeline([
		("estimator", estimator)
	])
	pipeline.fit(X_train, y_train)
	quality_train = quality(pipeline, X_train, y_train, target_field = y.name, data_usage = "training") \
		.with_all_metrics()
	quality_test = quality(pipeline, X_test, y_test, target_field = y.name, data_usage = "test") \
		.with_all_metrics()
	return ModelExplanation() \
		.append(Extension(name = "timestamp", value = datetime.datetime.utcnow().isoformat())) \
		.append(quality_train).append(quality_test)

class ModelExplanationTest(TestCase):
	
	def test_classifier(self):
		iris_X, iris_y = load_iris(return_X_y = True, as_frame = True)
		model_explanation = make_model_explanation(LogisticRegression(), iris_X, iris_y, ClassifierQuality)
		pmml = model_explanation.tostring()
		self.assertTrue(len(pmml) > 100)

	def test_binary_classifier(self):
		iris_X, iris_y = load_iris(return_X_y = True, as_frame = True)
		# Transform the label from multiclass to binary
		iris_y = (iris_y == 1).apply(lambda x: "yes" if x else "no")
		model_explanation = make_model_explanation(LogisticRegression(), iris_X, iris_y, BinaryClassifierQuality)
		pmml = model_explanation.tostring()
		self.assertTrue(len(pmml) > 100)

	def test_regressor(self):
		diabetes_X, diabetes_y = load_diabetes(return_X_y = True, as_frame = True)
		model_explanation = make_model_explanation(LinearRegression(), diabetes_X, diabetes_y, RegressorQuality)
		pmml = model_explanation.tostring()
		self.assertTrue(len(pmml) > 100)
