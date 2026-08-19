SkLearn2PMML [![Build Status](https://github.com/jpmml/sklearn2pmml/workflows/pytest/badge.svg)](https://github.com/jpmml/sklearn2pmml/actions?query=workflow%3A%22pytest%22)
============

Python package for converting [Scikit-Learn](https://scikit-learn.org/) pipelines to PMML.

# Features #

This package is a thin Python wrapper around the [JPMML-SkLearn](https://github.com/jpmml/jpmml-sklearn) library.

# News and Updates #

The current version is **0.133.0** (19 August, 2026):

```
pip install sklearn2pmml==0.133.0
```

See the [NEWS.md](https://github.com/jpmml/sklearn2pmml/blob/master/NEWS.md#01330) file.

# Prerequisites #

* Java 11 or newer. The Java executable must be available on system path.
* Python 3.8 or newer.

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

## Native Scikit-Learn ##

SkLearn2PMML can convert a wide variety of Scikit-Learn and Scikit-Learn adjacent estimators as-is.

The list of supported transformer, selector and predictor (aka model) classes is given in the [features.md](https://github.com/jpmml/jpmml-sklearn/blob/master/features.md) file of the JPMML-SkLearn project.

Keep SkLearn2PMML maximally up-to-date.
One and the same package version -- preferably the latest and greatest -- is able to work with all Scikit-Learn 0.17 (ca 2015) and newer versions.

### Library

Use the `sklearn2pmml.sklearn2pmml(estimator, pmml_path)` utility function to convert a fitted estimator object to PMML:

```python
from sklearn2pmml import sklearn2pmml

estimator = ...
estimator.fit(X, y)

# Convert a live estimator object
sklearn2pmml(estimator, "Estimator.pmml")
```

The `estimator` argument may also be a path-like object to an estimator pickle file in local filesystem:

```python
from sklearn2pmml import sklearn2pmml

import joblib

joblib.dump(estimator, "Estimator.pkl")

sklearn2pmml("Estimator.pkl", "Estimator.pmml")
```

SkLearn2PMML uses a custom Java component (rather than the built-in Python unpickler component) for reading pickle files.
As such, it is safe to use with unvetted pickle files.

### Command-line application

The `sklearn2pmml` module is executable.

The main application simply calls the `sklearn2pmml.sklearn2pmml()` utility function.
At minimum, it is necessary to provide the input pickle file (`-i` or `--input`; supports `joblib`, `pickle` or `dill` variants) and output PMML file paths (`-o` or `--output`):

```bash
python -m sklearn2pmml --input Estimator.pkl --output Estimator.pmml
```

To see all supported command-line options, pass `--help`:

```bash
python -m sklearn2pmml --help
```

On some platforms, the [Pip](https://pypi.org/project/pip/) package installer additionally makes the main application available as a top-level command:

```bash
sklearn2pmml --input pipeline.pkl --output pipeline.pmml
```

## PMML-enhanced Scikit-Learn ##

Native Scikit-Learn estimators have rather limited portability between environments, because they lack adequate metadata.
For example, they did not collect and store even the most crucial metadata about the feature matrix (ie. the `feature_names_in_` attribute) prior to Scikit-Learn 1.0 (ca 2021).

SkLearn2PMML provides the `sklearn2pmml.pipeline.PMMLPipeline` meta-estimator class, which extends the `sklearn.pipeline.Pipeline` class with the following functionality:

* Collect feature and label metadata using the `fit(X, y)` method:
  * The column names of the `X` dataset become input field names. Otherwise, they default to `x1`, `x2`, ..., `x{n_features_in_}`.
  * The column names of the `y` dataset become target field name(s). Otherwise, they default to `y` (single-output case) or `y1`, `y2`, ..., `y{n_outputs_}` (multi-output case).
* Perform prediction post-processing using `predict_transform(X)`, `predict_proba_transform(X)` and `apply_transform(X)` methods (operating on `predict_transformer`, `predict_proba_transformer` and `apply_transformer` attributes, respectively).
* Embed model verification data using the `verify(X)` method.
* Configure the representation of final estimator step using the `configure(**pmml_options)` method.
* Perform extra edits (ie. insert, update or delete PMML XML fragments) on the PMML document using the `customize(command, xpath_expr, pmml_element)` method.

PMML-enhanced workflow:

```python
#from sklearn.pipeline import Pipeline
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline

#pipeline = Pipeline(...)
# Activate prediction post-processing
pipeline = PMMLPipeline(..., predict_transformer = ...)
pipeline.fit(X, y)

# Embed small but representative sample for self-check purposes during deployment
pipeline.verify(X.sample(n = 10))

# Default prediction
yt = pipeline.predict(X)
# Default prediction, together with its transformation results
yt_transformed = pipeline.predict_transform(X)

# Default PMML representation
sklearn2pmml(pipeline, "Pipeline.pmml")

pipeline.configure(...)
#pipeline.customize(...)

# Customized PMML representation
sklearn2pmml(pipeline, "Pipeline-customized.pmml")
```

Additionally, SkLearn2PMML provides a number of PMML-oriented transformer, selector and predictor classes:

* `sklearn2pmml.decoration`. Capture or declare the domain of individual features by their operational type using `ContinuousDomain`, `CategoricalDomain` or `OrdinalDomain` meta-transformers. Give transformed features meaningful names using `Alias` and `MultiAlias` meta-transformers.
* `sklearn2pmml.preprocessing`. Transform features using `ExpressionTransformer` (any to any), `CutTransformer` (continuous to discrete), `LookupTransformer` (discrete to discrete), and many other transformers.
* `sklearn2pmml.cross_reference`. Cross-reference features and transformed features at subsequent transformer steps using `Memorizer` and `Recaller` meta-transformers.
* `sklearn2pmml.ensemble`. Estimate conditionally using the `SelectFirstTransformer` meta-transformer, plus `SelectFirstClassifier` and `SelectFirstRegressor` meta-predictors. Combine predictors using `GBDTLRClassifier` and `GBDTLMRegressor` meta-predictors.
* `sklearn2pmml.postprocessing`. Transform predictions using the `BusinessDecisionTransformer` transformer.

For example, mapping and pre-processing the [Audit](https://github.com/jpmml/jpmml-sklearn/blob/master/pmml-sklearn/src/test/resources/csv/Audit.csv) dataset:

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn2pmml.decoration import Alias, CategoricalDomain, ContinuousDomain
from sklearn2pmml.preprocessing import ExpressionTransformer

import pandas

df = pandas.read_csv("Audit.csv")

# Group features by type (operational type plus data type)
cat_cols = ["Education", "Employment", "Marital", "Occupation", "Gender"]
cont_int_cols = ["Age", "Hours"]
cont_float_cols = ["Income"]

transformer = ColumnTransformer([
	# Features
	("cat", make_pipeline(CategoricalDomain(), OneHotEncoder()), cat_cols),
	("cont_int", ContinuousDomain(), cont_int_cols),
	("cont_float", ContinuousDomain(), cont_float_cols),
	# Transformed features
	("hourly_income", Alias(ExpressionTransformer("X['Income'] / (X['Hours'] * 52)"), name = "Hourly_Income"), ["Income", "Hours"])
], remainder = "drop")
transformer.fit(df)

Xt = transformer.transform(df)
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

# License #

SkLearn2PMML is licensed under the terms and conditions of the [GNU Affero General Public License, Version 3.0](https://www.gnu.org/licenses/agpl-3.0.html).

If you would like to use SkLearn2PMML in a proprietary software project, then it is possible to enter into a licensing agreement which makes SkLearn2PMML available under the terms and conditions of the [BSD 3-Clause License](https://opensource.org/licenses/BSD-3-Clause) instead.

# Additional information #

SkLearn2PMML is developed and maintained by Openscoring Ltd, Estonia.

Interested in using [Java PMML API](https://github.com/jpmml) software in your company? Please contact [info@openscoring.io](mailto:info@openscoring.io)
