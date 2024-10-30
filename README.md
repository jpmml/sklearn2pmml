SkLearn2PMML [![Build Status](https://github.com/jpmml/sklearn2pmml/workflows/python/badge.svg)](https://github.com/jpmml/sklearn2pmml/actions?query=workflow%3A%22python%22)
============

Python package for converting [Scikit-Learn](https://scikit-learn.org/) pipelines to PMML.

# Features #

This package is a thin Python wrapper around the [JPMML-SkLearn](https://github.com/jpmml/jpmml-sklearn#features) library.

# News and Updates #

The current version is **0.111.1** (28 October, 2024):

```
pip install sklearn2pmml==0.111.1
```

See the [NEWS.md](https://github.com/jpmml/sklearn2pmml/blob/master/NEWS.md#01111) file.

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

Integrations:

* [Training Scikit-Learn GridSearchCV StatsModels pipelines](https://openscoring.io/blog/2023/10/15/sklearn_statsmodels_gridsearchcv_pipeline/)
* [Converting Scikit-Learn H2O.ai pipelines to PMML](https://openscoring.io/blog/2023/07/17/converting_sklearn_h2o_pipeline_pmml/)
* [Converting customized Scikit-Learn estimators to PMML](https://openscoring.io/blog/2023/05/03/converting_sklearn_subclass_pmml/)
* [Training Scikit-Learn StatsModels pipelines](https://openscoring.io/blog/2023/03/28/sklearn_statsmodels_pipeline/)
* [Upgrading Scikit-Learn XGBoost pipelines](https://openscoring.io/blog/2023/02/06/upgrading_sklearn_xgboost_pipeline_pmml/)
* [Training Python-based XGBoost accelerated failure time models](https://openscoring.io/blog/2023/01/28/python_xgboost_aft_pmml/)
* [Converting Scikit-Learn PyCaret 3 pipelines to PMML](https://openscoring.io/blog/2023/01/12/converting_sklearn_pycaret3_pipeline_pmml/)
* [Training Scikit-Learn H2O.ai pipelines](https://openscoring.io/blog/2022/11/11/sklearn_h2o_pipeline/)
* [One-hot encoding categorical features in Scikit-Learn XGBoost pipelines](https://openscoring.io/blog/2022/04/12/onehot_encoding_sklearn_xgboost_pipeline/)
* [Training Scikit-Learn TF(-IDF) plus XGBoost pipelines](https://openscoring.io/blog/2021/02/27/sklearn_tf_tfidf_xgboost_pipeline/)
* [Converting Scikit-Learn TF(-IDF) pipelines to PMML](https://openscoring.io/blog/2021/01/17/converting_sklearn_tf_tfidf_pipeline_pmml/)
* [Converting Scikit-Learn Imbalanced-Learn pipelines to PMML](https://openscoring.io/blog/2020/10/24/converting_sklearn_imblearn_pipeline_pmml/)
* [Converting logistic regression models to PMML](https://openscoring.io/blog/2020/01/19/converting_logistic_regression_pmml/#scikit-learn)
* [Stacking Scikit-Learn, LightGBM and XGBoost models](https://openscoring.io/blog/2020/01/02/stacking_sklearn_lightgbm_xgboost/)
* [Converting Scikit-Learn GridSearchCV pipelines to PMML](https://openscoring.io/blog/2019/12/25/converting_sklearn_gridsearchcv_pipeline_pmml/)
* [Converting Scikit-Learn TPOT pipelines to PMML](https://openscoring.io/blog/2019/06/10/converting_sklearn_tpot_pipeline_pmml/)
* [Converting Scikit-Learn LightGBM pipelines to PMML](https://openscoring.io/blog/2019/04/07/converting_sklearn_lightgbm_pipeline_pmml/)

Extensions:

* [Extending Scikit-Learn with feature cross-references](https://openscoring.io/blog/2023/11/25/sklearn_feature_cross_references/)
* [Extending Scikit-Learn with UDF expression transformer](https://openscoring.io/blog/2023/03/09/sklearn_udf_expression_transformer/)
* [Extending Scikit-Learn with CHAID models](https://openscoring.io/blog/2022/07/14/sklearn_chaid_pmml/)
* [Extending Scikit-Learn with prediction post-processing](https://openscoring.io/blog/2022/05/06/sklearn_prediction_postprocessing/)
* [Extending Scikit-Learn with outlier detector transformer](https://openscoring.io/blog/2021/07/16/sklearn_outlier_detector_transformer/)
* [Extending Scikit-Learn with date and datetime features](https://openscoring.io/blog/2020/03/08/sklearn_date_datetime_pmml/)
* [Extending Scikit-Learn with feature specifications](https://openscoring.io/blog/2020/02/23/sklearn_feature_specification_pmml/)
* [Extending Scikit-Learn with GBDT+LR ensemble models](https://openscoring.io/blog/2019/06/19/sklearn_gbdt_lr_ensemble/)
* [Extending Scikit-Learn with business rules model](https://openscoring.io/blog/2018/09/17/sklearn_business_rules/)

Miscellaneous:

* [Upgrading Scikit-Learn decision tree models](https://openscoring.io/blog/2023/12/29/upgrading_sklearn_decision_tree/)
* [Measuring the memory consumption of Scikit-Learn models](https://openscoring.io/blog/2022/11/09/measuring_memory_sklearn/)
* [Benchmarking Scikit-Learn against JPMML-Evaluator](https://openscoring.io/blog/2021/08/04/benchmarking_sklearn_jpmml_evaluator/)
* [Analyzing Scikit-Learn feature importances via PMML](https://openscoring.io/blog/2021/07/11/analyzing_sklearn_feature_importances_pmml/)

Archived:

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
