SkLearn2PMML [![Build Status](https://github.com/jpmml/sklearn2pmml/workflows/python/badge.svg)](https://github.com/jpmml/sklearn2pmml/actions?query=workflow%3A%22python%22)
============

Python package for converting [Scikit-Learn](https://scikit-learn.org/) pipelines to PMML.

# Features #

This package is a thin Python wrapper around the [JPMML-SkLearn](https://github.com/jpmml/jpmml-sklearn#features) library.

# Prerequisites #

* Java 1.8 or newer. The Java executable must be available on system path.
* Python 2.7, 3.4 or newer.

# Installation #

Installing a release version from PyPI:

```
pip install sklearn2pmml
```

Alternatively, installing the latest snapshot version from GitHub:

```
pip install --upgrade git+https://github.com/jpmml/sklearn2pmml.git
```

# Usage #

A typical workflow can be summarized as follows:

1. Create a `PMMLPipeline` object, and populate it with pipeline steps as usual. Class `sklearn2pmml.pipeline.PMMLPipeline` extends class `sklearn.pipeline.Pipeline` with the following functionality:
  * If the `PMMLPipeline.fit(X, y)` method is invoked with `pandas.DataFrame` or `pandas.Series` object as an `X` argument, then its column names are used as feature names. Otherwise, feature names default to "x1", "x2", .., "x{number_of_features}".
  * If the `PMMLPipeline.fit(X, y)` method is invoked with `pandas.Series` object as an `y` argument, then its name is used as the target name (for supervised models). Otherwise, the target name defaults to "y".
2. Fit and validate the pipeline as usual.
3. Optionally, compute and embed verification data into the `PMMLPipeline` object by invoking `PMMLPipeline.verify(X)` method with a small but representative subset of training data.
4. Convert the `PMMLPipeline` object to a PMML file in local filesystem by invoking utility method `sklearn2pmml.sklearn2pmml(pipeline, pmml_destination_path)`.

Developing a simple decision tree model for the classification of iris species:

```python
import pandas

iris_df = pandas.read_csv("Iris.csv")

iris_X = iris_df[iris_df.columns.difference(["Species"])]
iris_y = iris_df["Species"]

from sklearn.tree import DecisionTreeClassifier
from sklearn2pmml.pipeline import PMMLPipeline

pipeline = PMMLPipeline([
	("classifier", DecisionTreeClassifier())
])
pipeline.fit(iris_X, iris_y)

from sklearn2pmml import sklearn2pmml

sklearn2pmml(pipeline, "DecisionTreeIris.pmml", with_repr = True)
```

Developing a more elaborate logistic regression model for the same:

```python
import pandas

iris_df = pandas.read_csv("Iris.csv")

iris_X = iris_df[iris_df.columns.difference(["Species"])]
iris_y = iris_df["Species"]

from sklearn_pandas import DataFrameMapper
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn2pmml.decoration import ContinuousDomain
from sklearn2pmml.pipeline import PMMLPipeline

pipeline = PMMLPipeline([
	("mapper", DataFrameMapper([
		(["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"], [ContinuousDomain(), SimpleImputer()])
	])),
	("pca", PCA(n_components = 3)),
	("selector", SelectKBest(k = 2)),
	("classifier", LogisticRegression(multi_class = "ovr"))
])
pipeline.fit(iris_X, iris_y)
pipeline.verify(iris_X.sample(n = 15))

from sklearn2pmml import sklearn2pmml

sklearn2pmml(pipeline, "LogisticRegressionIris.pmml", with_repr = True)
```

# Documentation #

Up-to-date:

* [Extending Scikit-Learn with CHAID model type](https://openscoring.io/blog/2022/07/14/sklearn_chaid_pmml/)
* [Extending Scikit-Learn with prediction post-processing](https://openscoring.io/blog/2022/05/06/sklearn_prediction_postprocessing/)
* [One-hot-encoding (OHE) categorical features in Scikit-Learn based XGBoost pipelines](https://openscoring.io/blog/2022/04/12/onehot_encoding_sklearn_xgboost_pipeline/)
* [Benchmarking Scikit-Learn against JPMML-Evaluator in Java and Python environments](https://openscoring.io/blog/2021/08/04/benchmarking_sklearn_jpmml_evaluator/)
* [Extending Scikit-Learn with outlier detector transformer type](https://openscoring.io/blog/2021/07/16/sklearn_outlier_detector_transformer/)
* [Analyzing Scikit-Learn feature importances via PMML](https://openscoring.io/blog/2021/07/11/analyzing_sklearn_feature_importances_pmml/)
* [Training Scikit-Learn based TF(-IDF) plus XGBoost pipelines](https://openscoring.io/blog/2021/02/27/sklearn_tf_tfidf_xgboost_pipeline/)
* [Converting Scikit-Learn based TF(-IDF) pipelines to PMML documents](https://openscoring.io/blog/2021/01/17/converting_sklearn_tf_tfidf_pipeline_pmml/)
* [Converting Scikit-Learn based Imbalanced-Learn (imblearn) pipelines to PMML documents](https://openscoring.io/blog/2020/10/24/converting_sklearn_imblearn_pipeline_pmml/)
* [Extending Scikit-Learn with date and datetime features](https://openscoring.io/blog/2020/03/08/sklearn_date_datetime_pmml/)
* [Extending Scikit-Learn with feature specifications](https://openscoring.io/blog/2020/02/23/sklearn_feature_specification_pmml/)
* [Converting logistic regression models to PMML documents](https://openscoring.io/blog/2020/01/19/converting_logistic_regression_pmml/#scikit-learn)
* [Stacking Scikit-Learn, LightGBM and XGBoost models](https://openscoring.io/blog/2020/01/02/stacking_sklearn_lightgbm_xgboost/)
* [Converting Scikit-Learn hyperparameter-tuned pipelines to PMML documents](https://openscoring.io/blog/2019/12/25/converting_sklearn_gridsearchcv_pipeline_pmml/)
* [Extending Scikit-Learn with GBDT plus LR ensemble (GBDT+LR) model type](https://openscoring.io/blog/2019/06/19/sklearn_gbdt_lr_ensemble/)
* [Converting Scikit-Learn based TPOT automated machine learning (AutoML) pipelines to PMML documents](https://openscoring.io/blog/2019/06/10/converting_sklearn_tpot_pipeline_pmml/)
* [Converting Scikit-Learn based LightGBM pipelines to PMML documents](https://openscoring.io/blog/2019/04/07/converting_sklearn_lightgbm_pipeline_pmml/)
* [Extending Scikit-Learn with business rules (BR) model type](https://openscoring.io/blog/2018/09/17/sklearn_business_rules/)

Slightly outdated:

* [Converting Scikit-Learn to PMML](https://www.slideshare.net/VilluRuusmann/converting-scikitlearn-to-pmml)

# De-installation #

Uninstalling:

```
pip uninstall sklearn2pmml
```

# License #

SkLearn2PMML is licensed under the terms and conditions of the [GNU Affero General Public License, Version 3.0](https://www.gnu.org/licenses/agpl-3.0.html).

If you would like to use SkLearn2PMML in a proprietary software project, then it is possible to enter into a licensing agreement which makes SkLearn2PMML available under the terms and conditions of the [BSD 3-Clause License](https://opensource.org/licenses/BSD-3-Clause) instead.

# Additional information #

SkLearn2PMML is developed and maintained by Openscoring Ltd, Estonia.

Interested in using [Java PMML API](https://github.com/jpmml) software in your company? Please contact [info@openscoring.io](mailto:info@openscoring.io)
