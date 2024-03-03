# 0.103.3 #

## Breaking changes

* Refactored the `PMMLPipeline.customize(customizations: [str])` method into `PMMLPipeline.customize(command: str, xpath_expr: str, pmml_element: str)`.

This method may be invoked any number of times.
Each invocation appends a `sklearn2pmml.customization.Customization` object to the `pmml_customizations_` attribute of the final estimator step.

The `command` argument is one of SQL-inspired keywords `insert`, `update` or `delete` (to insert a new element, or to update or delete an existing element, respectively).
The `xpath_expr` is an XML Path (XPath) expression for pinpointing the action site. The XPath expression is evaluated relative to the main model element.
The `pmml_element` is a PMML fragment string.

For example, suppressing the secondary results by deleting the `Output` element:

``` python
pipeline = PMMLPipeline([
  ("classifier", ...)
])
pipeline.fit(X, y)
pipeline.customize(command = "delete", xpath_expr = "//:Output")
```

## New features

* Added `sklearn2pmml.metrics` module.

This module provides high-level `BinaryClassifierQuality`, `ClassifierQuality` and `RegressorQuality` pmml classes for the automated generation of [`PredictiveModelQuality`](https://dmg.org/pmml/v4-4-1/ModelExplanation.html#xsdElement_PredictiveModelQuality) elements for most common estimator types.

Refactoring the v0.103.0 code example:

``` python
from sklearn2pmml.metrics import ModelExplanation, RegressorQuality

pipeline = PMMLPipeline([
  ("regressor", ...)
])
pipeline.fit(X, y)

model_explanation = ModelExplanation()
predictive_model_quality = RegressorQuality(pipeline, X, y, target_field = y.name) \
  .with_all_metrics()
model_explanation.append(predictive_model_quality)

pipeline.customize(command = "insert", pmml_element = model_explanation.tostring())
```

* Added `sklearn2pmml.util.pmml` module.

## Minor improvements and fixes

* Added `EstimatorProxy.classes_` propery.

* Extracted `sklearn2pmml.configuration` and `sklearn2pmml.customization` modules.


# 0.103.2 #

## Breaking changes

* Refactored the `transform(X)` methods of SkLearn2PMML custom transformers to maximally preserve the original type and dimensionality of data containers.

For example, if the input to a single-column transformation is a Pandas' Series, and the nature of the transformation allows for it, then the output will also be a Pandas' Series.
Previously, the output was force-converted into a 2D Numpy array of shape `(n_samples, 1)`.

This change should go unnoticed for the majority of pipelines, as most Scikit-Learn transformers and estimators are quite lenient towards what they accept as input.
Any conflicts can be resolved by converting and/or reshaping the data container to a 2D Numpy array manually.

## New features

* Improved support for Pandas' categorical data type.

There is now a clear distinction between "proto" and "post" states of a data type object.
A "proto" object is a `category` string literal or an empty `pandas.CategoricalDtype` object.
A "post" object is fully initialized `pandas.CategoricalDtype` object that has been retrieved from some data container (typically, a training dataset).

* Added `ExpressionTransformer.dtype_` attribute.

A fitted `ExpressionTransformer` object now holds data type information using two attributes.
First, the `dtype` attribute holds the "proto" state - what was requested.
Second, the `dtype_` attribute holds the "post" state - what was actually found and delivered.

For example:

``` python
transformer = ExpressionTransformer(..., dtype = "category")
Xt = transformer.fit_transform(X, y)

# Prints "category" string literal
print(transformer.dtype)

# Prints pandas.CategoricalDtype object
print(transformer.dtype_)
print(transformer.dtype_.categories)
```

* Added `SeriesConstructor` meta-transformer.

This meta-transformer supersedes the `DataFrameConstructor` meta-transformer for single-column data container conversion needs.

## Minor improvements and fixes

* Added `ExpressionTransformer.fit_transform(X, y)` method.

* Added `DataFrameConstructor.get_feature_names_out()` and `SeriesConstructor.get_feature_names_out()` methods.

This makes these two meta-transformers compatible with Scikit-Learn's `set_output` API.


# 0.103.1 #

## Breaking changes

None.

## New features

* Added support for `pandas.CategoricalDtype` data type to the `DiscreteDomain` class and its subclasses.

It has been possible to set the `DiscreteDomain.dtype` parameter to a Pandas' categorical data type for quite some time.
However, up until this point, the JPMML-SkLearn library did not interact with this extra information in any way, because the valid value space (VVS) was constructed solely based on the `DiscreteDomain.data_values_` attribute.

The Pandas' categorical data type is not relevant in pure Scikit-Learn workflows.
However, it is indispensable for the proper representation of categorical features in LightGBM and XGBoost workflows.

Default usage (the VVS is learned automatically from the training dataset):

``` python
domain = CategoricalDomain(..., dtype = "category")
```

Advanced usage (the VVS is pre-defined):

``` python
vvs = [...]

# The DiscreteDomain.data_values parameter expects a list-like of list-likes, hence the double indexing syntax
domain = CategoricalDomain(..., data_values = [vvs], dtype = CategoricalDtype(categories = vvs))
```

See [SkLearn2PMML-411](https://github.com/jpmml/sklearn2pmml/issues/411)

## Minor improvements and fixes

* Fixed the invalid value replacement for the "as_missing" treatment.

This bug manifested itself in configurations where the `DiscreteDomain.missing_value_replacement` parameter was unset (meaning "leave as default missing value"), and the `DiscreteDomain.missing_values` parameter was set to a non-`None` value (meaning "the default missing value is <value>").

* Updated JPMML-LightGBM dependency.


# 0.103.0 #

## Breaking changes

None.

## New features

* Added `PMMLPipeline.customize(customizations)` method.

This method accepts one or more PMML fragment strings, which will be embedded into the main model element after all the automated PMML generation routines have been completed.
The customizations may replace existing elements, or define completely new elements.

The intended use case is defining model metadata such as [`ModelStats`](https://dmg.org/pmml/v4-4-1/Statistics.html#xsdElement_ModelStats) and [`ModelExplanation`](https://dmg.org/pmml/v4-4-1/ModelExplanation.html#xsdElement_ModelExplanation) elements.

For example, embedding regression model quality information:

``` python
from lxml import etree

pipeline = PMMLPipeline([
  ("regressor", ...)
])
pipeline.fit(X, y)

# Calculate R squared
score = pipeline.score(X, y)

# Generate a PMML 4.4 fragment
model_explanation = etree.Element("{http://www.dmg.org/PMML-4_4}ModelExplanation")
predictive_model_quality = etree.SubElement(model_explanation, "{http://www.dmg.org/PMML-4_4}PredictiveModelQuality")
predictive_model_quality.attrib["targetField"] = y.name
predictive_model_quality.attrib["r-squared"] = str(score)

pipeline.customize(etree.tostring(model_explanation))
```

See [SkLearn2PMML-410](https://github.com/jpmml/sklearn2pmml/issues/410)

## Minor improvements and fixes

* Fixed the scoping of target fields in `StackingClassifier` and `StackingRegressor` estimators.

See [JPMML-SkLearn-192](https://github.com/jpmml/jpmml-sklearn/issues/192)

* Updated all JPMML-Converter library dependencies to latest versions.


# 0.102.0 #

## Breaking changes

* Changed the default value of `Domain.with_statistics` attribute from `True` to `False`.

This attribute controls the calculation of descriptive statistics during the fitting.
The calculation of some descriptive statistics is costly (eg. interquartile range, median, standard deviation), which causes a notable flow-down of the `Domain.fit(X, y)` method.

The descriptive statistics about the training dataset is stored using the [`ModelStats`](https://dmg.org/pmml/v4-4-1/Statistics.html#xsdElement_ModelStats) element under the main model element (ie. the `/PMML/<Model>/ModelStats` elenment).
It is there for information purposes only. Its presence or absence does not affect the predictive capabilities of the model in any way.

## New features

* Fixed the `Domain.transform(X)` method to preserve the `X` argument unchanged.

If the decorator needs to modify the dataset in any way (eg. performing missing or invalid value replacement), then it will create a copy of the argument dataset before modifying it.
Otherwise, the argument dataset is passed through as-is.

This aligns decorators with Scikit-Learn API guidelines that transformers and transformer-likes should not tamper with the original dataset.

* Added support for One-Model-Per-Target (OMPT)-style multi-target XGBoost estimators.

When `XGBClassifier.fit(X, y)` and `XGBRegressor.fit(X, y)` methods are passed a multi-column `y` dataset, then XGBoost trains a OMPT-style multi-target model by default.

An OMPT-style multi-target model is functionally identical to a collection of single-target models, as all targets are handled one-by-one both during fitting and prediction.
In other words, the use of `MultiOutputClassifier` and `MultiOutputRegressor` meta-estimators is now deprecated when modelling multi-target datasets with XGBoost estimators.

Before:

``` python
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

X = ...
# A multi-column 2D array
ynd = ...

regressor = MultiOutputRegressor(XGBRegressor())
regressor.fit(X, ynd)
```

After: 

``` python
regressor = XGBRegressor()
regressor.fit(X, ynd)
```

* Ensured XGBoost 2.0 compatibility:
  * Improved the partitioning of the main trees array into sub-arrays based on model type (boosting vs. bagging) and target cardinality (single-target vs. multi-target).
  * Improved support for early stopping.

See [JPMML-XGBoost v1.8.2](https://github.com/jpmml/jpmml-xgboost/blob/master/NEWS.md#182)

Earlier SkLearn2PMML package versions may accept and convert XGBoost 2.0 without errors, but the resulting PMML document may contain an ensemble model with a wrong selection and/or wrong number of member decision tree models in it.
These kind of conversion issues can be easily detected by embedding the model verification dataset into the model.

## Minor improvements and fixes

* Improved support for `XGBClassifier.classes_` property.

This member was promoted from attribute to property during the XGBoost 1.7 to 2.0 upgrade, thereby making it "invisible" in non-Python environments.

The temporary workaround was to manually re-assign this property to a `XGBClassifier.pmml_classes_` attribute.
While the above workaround continues to be relevant with advanced targets (eg. string-valued category levels) it is no longer needed for default targets.

See [SkLearn2PMML-402](https://github.com/jpmml/sklearn2pmml/issues/402)

* Added `GBDTLRClassifier.classes_` property.


# 0.101.0 #

## Breaking changes

* Renamed the `DiscreteDomain.data_` attribute to `data_values_`.

## New features

* Added support for multi-column mode to the `DiscreteDomain` class and its subclasses (`CategoricalDomain` and `OrdinalDomain`).

This brings discrete decorators to functional parity with continuous decorators, which have been supporting both single-column and multi-column mode for years.

Before:

``` python
from sklearn_pandas import DataFrameMapper
from sklearn2pmml.decoration import CategoricalDomain, ContinuousDomain

cat_cols = [...]
cont_cols = [...]

mapper = DataFrameMapper(
  # Map continuous columns in one go
  [([cont_cols], ContinuousDomain(...))] +
  # Map categorical columns one by one
  [([cat_col], CategoricalDomain(...)) for cat_col in cat_cols]
)
```

After:

``` python
mapper = DataFrameMapper([
  # Map both continuous and categorical columns in one go
  ([cont_cols], ContinuousDomain(...)),
  ([cat_cols], CategoricalDomain(...))
])
```

* Added support for user-defined valid value spaces:
  * `ContinuousDomain.data_min` and `ContinuousDomain.data_max` parameters (scalars, or a list-like of scalars depending on the multiplicity).
  * The `DiscreteDomain.data_values` parameter (a list-like, or a list-like of list-likes depending on multiplicity).

This allows the data scientist to specify valid value spaces that are different (typically, wider) than the valid value space that can be inferred from the training dataset during the fitting.

Extending the valid value space for the "iris" dataset:

``` python
from sklearn.datasets import load_iris

iris_X, iris_y = load_iris(return_X_y = True, as_frame = True)

columns = iris_X.columns.values

# Extend all feature bounds to [0.0 .. 10.0]
data_usermin = [0.0] * len(columns)
data_usermax = [10.0] * len(columns)

mapper = DataFrameMapper([
  (columns.tolist(), ContinuousDomain(data_min = data_usermin, data_max = data_usermax))
])
mapper.fit_transform(iris_X, iris_y)
```

* Improved support for the "category" data type in the `CastTransformer.fit(X, y)` method.

If the `CastTransformer.dtype` parameter value is "category" (ie. a string literal), then the fit method will auto-detect valid category levels, and will set the `CastTransformer.dtype_` attribute to a `pandas.CategoricalDtype` object instead.
The subsequent transform method invocations are now guaranteed to exhibit stable transformation behaviour.
Previously, each method call was computing its own set of valid category values.

* Added the `Decorator` class to the `sklearn.base.OneToOneFeatureMixin` class hierarchy.

This makes decorators compatible with Scikit-Learn's `set_output` API.

Choosing a data container for transformation results:

``` python
from sklearn.compose import ColumnTransformer

transformer = ColumnTransformer([
  ("cont", ContinuousDomain(...), cont_cols),
  ("cat", CategoricalDomain(...), cat_cols)
])

# Force the output data container to be a Pandas' DataFrame (rather than a Numpy array)
transformer.set_output(transform = "pandas")
```

* Added `CastTransformer` and `IdentityTransformer` classes to the `sklearn.base.OneToOneFeatureMixin` class hierarchy.

This makes these two transformers compatible with Scikit-Learn's `set_output` API.

* Added `Memorizer.get_feature_names_out()` and `Recaller.get_feature_names_out()` methods.

This makes memory managers compatible with Scikit-Learn's `set_output` API.

## Minor improvements and fixes

* Updated formal package requirements to `scikit-learn >= 1.0`, `numpy >= 1.22(.4)` and `pandas >= 1.5(.3)`.

* Optimized `ContinuousDomain.fit(X, y)` and `DiscreteDomain.fit(X, y)` methods.

* Stopped auto-adding the `DiscreteDomain.missing_value_replacement` parameter value into the valid value space of discrete domains.

The missing value replacement value should occur in the training set naturally. If not, it would be more appropriate to manually define the valid value space using the newly introduced `DiscreteDomain.data_values` parameter.

* Improved handling of missing values in the `CastTransformer.fit(X, y)` method.

Previously, it was possible that the `float("NaN")` value could be included into the list of valid category levels when casting sparse string columns to the categorical data type.

* Added `sklearn2pmml.util.to_numpy(X)` utility function.
