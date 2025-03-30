# 0.116.1 #

## Breaking changes

* Refactored the cast to Python string data types (`str` and `"unicode"`).

Previously, the cast was implemented using data container-native `apply` methods (eg. `X.astype(str)`). However, these methods act destructively with regards to constant values such as `None`, `numpy.nan`, `pandas.NA` and `pandas.NaT`, by replacing them with the corresponding string literals. For example, a `None` constant becomes a `"None"` string.

The `sklearn2pmml.util.cast()` utility function that implements casts across SkLearn2PMML transformers now contains an extra masking logic to detect and preserve missing value constants unchanged. This is critical for the correct functioning of downstream missing value-aware steps such as imputers, encoders and expression transformers.

See [SkLearn2PMML-445](https://github.com/jpmml/sklearn2pmml/issues/445)

## New features

* Added `LagTransformer.block_indicators` and `RollingAggregateTransformer.block_indicators` attributes.

These attributes enhance the base transformation with "group by" functionality.

For example, calculating a moving average over a mixed stock prices dataset:

``` python
from sklearn2pmml.preprocessing import RollingAggregateTransformer

mapper = DataFrameMapper([
  (["stock", "price"], RollingAggregateTransformer(function = "avg", n = 100, block_indicators = ["stock"]))
], input_df = True, df_out = True)
```

* Added package up-to-date check.

The Java side of the package computes the timedelta between the current timestamp and the package build timestamp before doing any actual work. If this timedelta is greater than 180 days (6 months) a warning is issued. If this timedelta is greater than 360 days (12 months) an error is raised.

## Minor improvements and fixes

* Added `LagTransformer.get_feature_names_out()` and `RollingAggregateTransformer.get_feature_names_out()` methods.

* Fixed the cast of wildcard features.

Previously, if a `CastTransformer` transformer was applied to a wildcard feature, then the newly assigned operational type was not guaranteed to stick.

See [SkLearn2PMML-445](https://github.com/jpmml/sklearn2pmml/issues/445#issuecomment-2737638764)


# 0.116.0 #

## Breaking changes

* Renamed `sklearn2pmml.preprocessing.Aggregator` class to `AggregateTransformer`.

In order to support archived pipeline objects, the SkLearn2PMML package shall keep recognizing the old name variant alongside the new one.

## New features

* Added support for [`sklearn.model_selection.FixedThresholdClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.FixedThresholdClassifier.html) and [`sklearn.model_selection.TunedThresholdClassifierCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TunedThresholdClassifierCV.html) classes.

The post-fit tuned target is exposed in the model schema as an extra `thresholded(<target field name>)` output field.

* Added support for `sklearn2pmml.preprocessing.LagTransformer` class.

Implements a "shift" operation using PMML's [`Lag`](https://dmg.org/pmml/v4-4-1/Transformations.html#lag) element.

* Added support for `sklearn2pmml.preprocessing.RollingAggregateTransformer` class.

Implements a "rolling aggregate" operation using PMML's [`Lag`](https://dmg.org/pmml/v4-4-1/Transformations.html#lag) element.

The PMML implementation differs from Pandas' default implementation in that it excludes the curent row. For example, when using a window size of five, then PMML considers five rows preceding the current row (ie. `X.rolling(window = 5, closed = "left")`), whereas Pandas considers four rows preceding the current row plus the current row (ie. `X.rolling(window = 5, closed = "right")`).

A Pandas-equivalent "rolling aggregate" operation can be emulated using `AggregateTransformer` and `LagTransformer` transformers directly.

## Minor improvements and fixes

None.


# 0.115.0 #

## Breaking changes

None.

## New features

* Added support for `sklearn2pmml.preprocessing.StringLengthTransformer` class.

## Minor improvements and fixes

* Fixed the `StringNormalizer.transform(X)` method to preserve the original data container shape.

See [SkLearn2PMML-443](https://github.com/jpmml/sklearn2pmml/issues/443)

* Ensured compatibility with PCRE2 0.5.0.

The 0.5.X development branch underwent breaking changes, with the goal of migrating from proprietary API to Python RE-compatible API. For example, the compiler pattern object now provides both `search(x)` and `sub(replacement, x)` conveniene methods.

* Ensured compatibility with BorutaPy 0.4.3, Category-Encoders 2.6.4, CHAID 5.4.2, Hyperopt-sklearn 1.0.3, Imbalanced-Learn 0.13.0, InterpretML 0.6.9, OptBinning 0.20.1, PyCaret 3.3.2, Scikit-Lego 0.9.4, Scikit-Tree 0.8.0 and TPOT 0.12.2.


# 0.114.0 #

## Breaking changes

* Required Java 11 or newer.

## New features

None.

## Minor improvements and fixes

None.


# 0.113.0 #

## Breaking changes

None.

## New features

None.

## Minor improvements and fixes

* Updated Java libraries.


# 0.112.1 #

## Breaking changes

None.

## New features

* Added support for in-place file conversion.

If the `estimator` parameter to the `sklearn2pmml.sklearn2pmml(estimator, pmml_path)` utility function is a path-like object (eg. `pathlib.Path` or string), then the Pytho side of the SkLearn2PMML package shall pass it forward to the Java side (without making any efforts to load or modify anything about it).

This opens the door for the safe conversion of legacy and/or potentially harmful Pickle files.

For example, attempting to convert an unknown origin and composition estimator file to a PMML document:

``` python
from sklearn2pmml import sklearn2pmml

sklearn2pmml("/path/to/estimator.pkl", "estimator.pmml")
```

## Minor improvements and fixes

* Added `--version` command-line option.

Checking the version of the currently installed command-line application:

```
sklearn2pmml --version
```

* Fixed the version standardization transformation.


# 0.112.0 #

## Breaking changes

* Required Python 3.8 or newer.

This requirement stems from underlying package requirements, most notably that of the NumPy package (`numpy>=1.24`).

Portions of the SkLearn2PMML package are still usable with earlier Python versions.
For example, the `sklearn2pmml.sklearn2pmml(estimator, pmml_path)` utlity function should work with any Python 2.7, 3.4 or newer version.

* Migrated setup from `distutils` to `setuptools`.

* Migrated unit tests from `nose` to `pytest`.

Testing the (source checkout of-) package:

```
python -m pytest .
```

## New features

* Added command-line interface to the `sklern2pmml.sklearn2pmml()` utility function.

Sample usage:

```
python -m sklearn2pmml --input pipeline.pkl --output pipeline.pmml
```

Getting help:

```
python -m sklearn2pmml --help
```

* Added `sklearn2pmml` command-line application.

Sample usage:

```
sklearn2pmml -i pipeline.pkl -o pipeline.pmml
```

## Minor improvements and fixes

None.


# 0.111.2 #

## Breaking changes

None.

## New features

* Separated version transformation into two parts - version standardization (from vendor-extended PMML 4.4 to standard PMML 4.4) and version downgrade (from PMML 4.4 to any earlier PMML version).

## Minor improvements and fixes

* Eliminated the use of temporary file(s) during version transformation.

* Improved version downgrade.


# 0.111.1 #

## Breaking changes

* Refactored the downgrading of PMML schema versions.

Previously, the downgrade failed if the generated PMML document was not strictly compatible with the requested PMML schema version. Also, the downgrade failed if there were any vendor extension attributes or elements around (ie. attributes prefixed with `x-` or elements prefixed with `X-`).

The new behaviour is to allow the downgrade to run to completion, and display a grave warning (together with the full list of incompatibilities) in the end.

See [SkLearn2PMML-433](https://github.com/jpmml/sklearn2pmml/issues/433#issuecomment-2442652934)

* Updated logging configuration.

The Java backend used to employ SLF4J's default logging configuration, which prints two lines per logging event - the first line being metadata, and the second line being the actual data.

The new logging configuration prints one line per logging event.
The decision was to drop the leading metadata line in order to de-clutter the console.

## New features

* Added support for using `pcre2` module functions in expressions and predicates.

For example, performing text replacement operation on a string column:

``` python
from sklearn2pmml.preprocessing import ExpressionTransformer

# Replace sequences of one or more 'B' characters with a single 'c' character
transformer = ExpressionTransformer("pcre2.substitute('B+', 'c', X[0])")
```

## Minor improvements and fixes

* Added support for single-quoted multiline strings in expressions and predictions.


# 0.111.0 #

## Breaking changes

* Assume `re` as the default regular expression (RE) flavour.

* Removed support for multi-column mode from `StrngNormalizer` class.
String transformations are unique and rare enough, so that they should be specified on a column-by-column basis.

## New features

* Added `MatchesTransformer.re_flavour` and `ReplaceTransformer.re_flavour` attributes.
The Python environment allows to choose between different RE engines, which vary by RE syntax to a material degree.
Unambiguous identification of the RE engine improves the portability of RE transformers between applications (train vs. deployment) and environments.

Supported RE flavours:

| RE flavour | Implementation |
|---|---|
| `pcre` | [PCRE](https://pypi.org/project/python-pcre/) package |
| `pcre2`| [PCRE2](https://pypi.org/project/pcre2/) package |
| `re` | Built-in `re` module |

PMML only supports Perl Compatible Regular Expression (PCRE) syntax.

It is recommended to use some PCRE-based RE engine on Python side as well to minimize the chance of "communication errors" between Python and PMML environments.

* Added `sklearn2pmml.preprocessing.regex.make_regex_engine(pattern, re_flavour)` utility function.

This utility function pre-compiles and wraps the specified RE pattern into a `sklearn2pmml.preprocessing.regex.RegExEngine` object.

The `RegExEngine` class provides `matches(x)` and `replace(replacement, x)` methods, which correspond to PMML's [`matches`](https://dmg.org/pmml/v4-4-1/BuiltinFunctions.html#matches) and [`replace`](https://dmg.org/pmml/v4-4-1/BuiltinFunctions.html#replace) built-in functions, respectively.

For example, unit testing a RE engine:

``` python
from sklearn2pmml.preprocessing.regex import make_regex_engine

regex_engine = make_regex_engine("B+", re_flavour = "pcre2")

assert regex_engine.matches("ABBA") == True
assert regex_engine.replace("c", "ABBA") == "AcA"
```

See [SkLearn2PMML-228](https://github.com/jpmml/sklearn2pmml/issues/228)

* Refactored `StringNormalizer.transform(X)` and `SubstringTransformer.transform(X)` methods to support Pandas' Series input and output.

See [SkLearn2PMML-434](https://github.com/jpmml/sklearn2pmml/issues/434)

## Minor improvements and fixes

* Ensured compatibility wth Scikit-Learn 1.5.1 and 1.5.2.


# 0.110.0 #

## Breaking changes

None.

## New features

* Added `pmml_schema` parameter to the `sklearn2pmml.sklearn2pmml(estimator, pmml_path)` utility function.

This parameter allows downgrading PMML schema version from the default 4.4 version to any 3.X or 4.X version.
However, the downgrade is "soft", meaning that it only succeeds if the in-memory PMML document is naturally compatible with the requested PMML schema version.
The downgrade fails if there are structural changes needed.

Exprting a pipeline into a PMML schema version 4.3 document:

``` python
from sklearn2pmml import sklearn2pmml

pipeline = Pipeline([...])
pipeline.fit(X, y)

sklearn2pmml(pipeline, "Pipeline-v43.pmml", pmml_schema = "4.3")
```

See [SkLearn2PMML-423](https://github.com/jpmml/sklearn2pmml/issues/423#issuecomment-2264552688)

Complex downgrades will be implemented based on customer needs.

## Minor improvements and fixes

None.


# 0.109.0 #

## Breaking changes

None.

## New features

* Added support for Scikit-Learn 1.5.X.

* Added support for `yeo-johnson` power transform method in [`PowerTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html) class.

This method is the default for this transformer.

## Minor improvements and fixes

* Fixed the initialization of Python expression evaluation environments.

The environment is "pre-loaded" with a small number of Python built-in (`math`, `re`) and third-party (`numpy`, `pandas`, `scipy` and optionally `pcre`) module imports.

All imports use canonical module names (eg. `import numpy`). There is **no** module name aliasing taking place (eg. `import numpy as np`).
Therefore, the evaluatable Python expressions must also spell out canonical module names.

See [SkLearn2PMML-421](https://github.com/jpmml/sklearn2pmml/issues/421)

* Added support for `log` link function in `ExplainableBoostingRegressor` class.

See [SkLearn2PMML-422](https://github.com/jpmml/sklearn2pmml/issues/422)


# 0.108.0 #

## Breaking changes

None.

## New features

* Added support for [`interpret.glassbox.ClassificationTree`](https://interpret.ml/docs/python/api/ClassificationTree.html) and [`interpret.glassbox.RegressionTree`](https://interpret.ml/docs/python/api/RegressionTree.html) classes.

* Added support for [`interpret.glassbox.LinearRegression`](https://interpret.ml/docs/python/api/LinearRegression.html) and [`interpret.glassbox.LogisticRegression`](https://interpret.ml/docs/python/api/LogisticRegression.html) classes.

* Added support for [`interpret.glassbox.ExplainableBoostingClassifier`](https://interpret.ml/docs/python/api/ExplainableBoostingClassifier.html) and [`interpret.glassbox.ExplainableBoostingRegressor`](https://interpret.ml/docs/python/api/ExplainableBoostingRegressor.html) classes.

See [InterpretML-536](https://github.com/interpretml/interpret/issues/536)

## Minor improvements and fixes

* Ensured compatibility with Scikit-Learn 1.4.2.


# 0.107.1 #

## Breaking changes

None.

## New features

* Added support for [`H2OExtendedIsolationForestEstimator`](https://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/modeling.html#h2oextendedisolationforestestimator) class.

This class implements the isolation forest algorithm using oblique tree models.
It is claimed to outperform the [`H2OIsolationForestEstimator`](https://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/modeling.html#h2oisolationforestestimator) class, which does the same using plain (ie. non-oblique) tree models.

* Made `lightgbm.Booster` class directly exportable to PMML.

The SkLearn2PMML package now supports both LightGBM [Training API](https://lightgbm.readthedocs.io/en/latest/Python-API.html#training-api) and [Scikit-Learn API](https://lightgbm.readthedocs.io/en/latest/Python-API.html#scikit-learn-api):

``` python
from lightgbm import train, Dataset
from sklearn2pmml import sklearn2pmml

ds = Dataset(data = X, label = y)

booster = train(params = {...}, train_set = ds)

sklearn2pmml(booster, "LightGBM.pmml")
```

* Made `xgboost.Booster` class directly exportable to PMML.

The SkLearn2PMML package now supports both XGBoost [Learning API](https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.training) and [Scikit-Learn API](https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn):

``` python
from xgboost import train, DMatrix
from sklearn2pmml import sklearn2pmml

dmatrix = DMatrix(data = X, label = y)

booster = train(params = {...}, dtrain = dmatrix)

sklearn2pmml(booster, "XGBoost.pmml")
```

* Added `xgboost.Booster.fmap` attribute.

This attribute allows overriding the embedded feature map with a user-defined feature map.

The main use case is refining the category levels of categorical levels.

A suitable feature map object can be generated from the training dataset using the `sklearn2pmml.xgboost.make_feature_map(X)` utility function:

``` python
from xgboost import train, DMatrix
from sklearn2pmml.xgboost import make_feature_map

# Enable categorical features
dmatrix = DMatrix(X, label = y, enable_categorical = True)

# Generate a feature map with detailed description of all continuous and categorical features in the dataset
fmap = make_feature_map(X)

booster = train(params = {...}, dtrain = dmatrix)
booster.fmap = fmap
```

* Added `input_float` conversion option for XGBoost models.

## Minor improvements and fixes

None.


# 0.107.0 #

## Breaking changes

None.

## New features

* Added support for [`sktree.ensemble.ExtendedIsolationForest`](https://docs.neurodata.io/scikit-tree/dev/generated/sktree.ExtendedIsolationForest.html) class.

For example, training and exporting an `ExtendedIsolationForest` outlier detector into a PMML document:

``` python
from sklearn.datasets import load_iris
from sktree.ensemble import ExtendedIsolationForest
from sklearn2pmml import sklearn2pmml

iris_X, iris_y = load_iris(return_X_y = True, as_frame = True)

eif = ExtendedIsolationForest(n_estimators = 13)
eif.fit(iris_X)

sklearn2pmml(eif, "ExtendedIsolationForestIris.pmml")
```

See [SKTree-255](https://github.com/neurodata/scikit-tree/issues/255)

* Added support for [`sktree.ensemble.ObliqueRandomForestClassifier`](https://docs.neurodata.io/scikit-tree/dev/generated/sktree.ObliqueRandomForestClassifier.html) and [`sktree.ensemble.ObliqueRandomForestRegressor`](https://docs.neurodata.io/scikit-tree/dev/generated/sktree.ObliqueRandomForestRegressor.html) classes.

* Added support for [`sktree.tree.ObliqueDecisionTreeClassifier`](https://docs.neurodata.io/scikit-tree/dev/generated/sktree.tree.ObliqueDecisionTreeClassifier.html) and [`sktree.tree.ObliqueDecisionTreeRegressor`](https://docs.neurodata.io/scikit-tree/dev/generated/sktree.tree.ObliqueDecisionTreeRegressor.html) classes.

## Minor improvements and fixes

None.


# 0.106.0 #

## Breaking changes

* Upgraded JPMML-SkLearn library from 1.7(.56) to 1.8(.0).

This is a major API upgrade.
The 1.8.X development branch is already source and binary incompatible with earlier 1.5.X through 1.7.X development branches, with more breaking changes to follow suit.

Custom SkLearn2PMML plugins would need to be upgraded and rebuilt.

## New features

None.

## Minor improvements and fixes

* Ensured compatibility with Python 3.12.

* Ensured compatibility with Dill 0.3.8.


# 0.105.2 #

## Breaking changes

None.

## New features

None.

## Minor improvements and fixes

* Improved support for categorical encoding over mixed datatype column sets.

Scikit-Learn transformers such as `OneHotEncoder`, `OrdinalEncoder` and `TargetEncoder` can be applied to several columns in one go.
Previously it was assumed that all columns shared the same data type. If that was assumption was violated in practice, they were all force cast to the `string` data type.

The JPMML-SkLearn library now detects and maintains the data type on a single column basis.

* Made Category-Encoders classes directly exportable to PMML.

For example, training and exporting a `BaseNEncoder` transformer into a PMML document for manual analysis and interpretation purposes:

``` python
from category_encoders import BaseNEncoder
from sklearn2pmml import sklearn2pmml

transformer = BaseNEncoder(base = 3)
transformer.fit(X, y = None)

sklearn2pmml(transformer, "Base3Encoder.pmml")
```

* Fixed support for `(category_encoders.utils.)BaseEncoder.feature_names_in_` attribute.

According to [SLEP007](https://scikit-learn-enhancement-proposals.readthedocs.io/en/latest/slep007/proposal.html), the value of a `feature_names_in_` attribute should be an array of strings.

Category-Encoders transformers are using a list of strings instead.

* Refactored `ExpressionClassifier` and `ExpressionRegressor` constructors.

The evaluatable object can now also be a string literal.


# 0.105.1 #

## Breaking changes

None.

## New features

* Added support for [`sklearn.preprocessing.TargetEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html) class.

* Added support for [`sklearn.preprocessing.SplineTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.SplineTransformer.html) class.

The `SplineTransformer` class computes a B-spline for a feature, which is then used to expand the feature into new features that correspond to B-spline basis elements.

This class is not suitable for simple feature and prediction scaling purposes (eg. calibration of computer probabilities).
Consider using the `sklearn2pmml.preprocessing.BSplineTransformer` class in such a situation.

* Added support for [`statsmodels.api.QuantReg`](https://www.statsmodels.org/dev/generated/statsmodels.regression.quantile_regression.QuantReg.html) class.

* Added `input_float` conversion option.

Scikit-Learn tree and tree ensemble models prepare their inputs by first casting them to `(numpy.)float32`, and then to `(numpy.)float64` (exactly so, even if the input value already happened to be of `(numpy.)float64` data type).

PMML does not provide effective means for implementing "chained casts"; the chain must be broken down into elementary cast operations, each of which is represented using a standalone `DerivedField` element.
For example, preparing the "Sepal.Length" field of the iris dataset:

``` xml
<PMML>
  <DataDictionary>
    <DataField name="Sepal.Length" optype="continuous" dataType="double">
      <Interval closure="closedClosed" leftMargin="4.3" rightMargin="7.9"/>
    </DataField>
  </DataDictionary>
  <TransformationDictionary>
    <DerivedField name="float(Sepal.Length)" optype="continuous" dataType="float">
      <FieldRef field="Sepal.Length"/>
    </DerivedField>
    <DerivedField name="double(float(Sepal.Length))" optype="continuous" dataType="double">
      <FieldRef field="float(Sepal.Length)"/>
    </DerivedField>
  </TransformationDictionary>
</PMML>
```

Activating the `input_float` conversion option:

``` python
pipeline = PMMLPipeline([
  ("classifier", DecisionTreeClassifier())
])
pipeline.fit(iris_X, iris_y)

# Default mode
pipeline.configure(input_float = False)
sklearn2pmml("DecisionTree-default.pmml")

# "Input float" mode
pipeline.configure(input_float = True)
sklearn2pmml("DecisionTree-input_float.pmml")
```

This conversion option updates the data type of the "Sepal.Length" data field from `double` to `float`, thereby eliminating the need for the first `DerivedField` element of the two:

``` xml
<PMML>
  <DataDictionary>
    <DataField name="Sepal.Length" optype="continuous" dataType="float">
      <Interval closure="closedClosed" leftMargin="4.300000190734863" rightMargin="7.900000095367432"/>
    </DataField>
  </DataDictionary>
  <TransformationDictionary>
    <DerivedField name="double(Sepal.Length)" optype="continuous" dataType="double">
      <FieldRef field="Sepal.Length"/>
    </DerivedField>
  </TransformationDictionary>
</PMML>
```

Changing the data type of a field may have side effects if the field contributes to more than one feature.
The effectiveness and safety of configuration options should be verified by integration testing.

* Added `H2OEstimator.pmml_classes_` attribute.

This attribute allows customizing target category levels.
It comes in handly when working with ordinal targets, where the H2O.ai framework requires that target category levels are encoded from their original representation to integer index representation.

A fitted H2O.ai ordinal classifier predicts integer indices, which must be manually decoded in the application layer.
The JPMML-SkLearn library is able to "erase" this encode-decode helper step from the workflow, resulting in a clean and efficient PMML document:

``` python
ordinal_classifier = H2OGeneralizedLinearEstimator(family = "ordinal")
ordinal_classifier.fit(...)

# Customize target category levels
# Note that the default lexicographic ordering of labels is different from their intended ordering
ordinal_classifier.pmml_classes_ = ["bad", "poor", "fair", "good", "excellent"]

sklearn2pmml(ordinal_classifier, "OrdinalClassifier.pmml")
```

## Minor improvements and fixes

* Fixed the categorical encoding of missing values.

This bug manifested itself when the input column was mixing different data type values.
For example, a sparse string column, where non-missing values are strings, and missing values are floating-point `numpy.NaN` values.

Scikit-Learn documentation warns against mixing string and numeric values within a single column, but it can happen inadvertently when reading a sparse dataset into a Pandas' DataFrame using standard library functions (eg. the [`pandas.read_csv()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html) function).

* Added Pandas to package dependencies.

See [SkLearn2PMML-418](https://github.com/jpmml/sklearn2pmml/issues/418)

* Ensured compatibility with H2O.ai 3.46.0.1.

* Ensured compatibility with BorutaPy 0.3.post0 (92e4b4e).


# 0.105.0 #

## Breaking changes

None.

## New features

* Added `Domain.n_features_in_` and `Domain.feature_names_in_` attributes.

This brings domain decorators to conformance with "physical" Scikit-Learn input inspection standards such as [SLEP007](https://scikit-learn-enhancement-proposals.readthedocs.io/en/latest/slep007/proposal.html) and [SLEP010](https://scikit-learn-enhancement-proposals.readthedocs.io/en/latest/slep010/proposal.html).

Domain decorators are natively about "logical" input inspection (ie. establishing and enforcing model's applicability domain).

By combining these two complementary areas of functionality, they now make a great **first** step for any pipeline:

``` python
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn2pmml.decoration import ContinuousDomain

iris_X, iris_y = load_iris(return_X_y = True, as_frame = True)

pipeline = Pipeline([
  # Collect column-oriented model's applicability domain
  ("domain", ContinuousDomain()),
  ("classifier", ...)
])
pipeline.fit(iris_X, iris_y)

# Dynamic properties, delegate to (the attributes of-) the first step
print(pipeline.n_features_in_)
print(pipeline.feature_names_in_)
```

* Added `MultiDomain.n_features_in_` and `MultiDomain.feature_names_in_` attribute.

* Added support for missing values in tree and tree ensemble models.

Scikit-Learn 1.3 extended the `Tree` data structure with a `missing_go_to_left` field.
This field indicates the default split direction for each split, and is always present and populated whether the training dataset actually contained any missing values or not.

As a result, Scikit-Learn 1.3 tree models are able to accept and make predictions on sparse datasets, even if they were trained on a fully dense dataset.
There is currently no mechanism for a data scientist to tag tree models as "can or cannot be used with missing values".

The JPMML-SkLearn library implements two `Tree` data structure conversion modes, which can be toggled using the `allow_missing` conversion option.
The default mode corresponds to Scikit-Learn 0.18 through 1.2 behaviour, where a missing input causes the evaluation process to immediately bail out with a missing prediction.
The "missing allowed" mode corresponds to Scikit-Learn 1.3 and newer behaviour, where a missing input is ignored, and the evaluation proceeds to the pre-defined child branch until a final non-missing prediction is reached.

Right now, the data scientist must activate the latter mode manually, by configuring `allow_missing = True`:

``` python
from sklearn.tree import DecisionTreeClassifier
from sklearn2pmml.pipeline import PMMLPipeline

pipeline = PMMLPipeline([
  ("classifier", DecisionTreeClassifier())
])
pipeline.fit(X, y)

# Default mode
pipeline.configure(allow_missing = False)
sklearn2pmml(pipeline, "DecisionTree-default.pmml")

# "Missing allowed" mode
pipeline.configure(allow_missing = True)
sklearn2pmml(pipeline, "DecisionTree-missing_allowed.pmml")
```

Both conversion modes generate standard PMML markup.
However, the "missing allowed" mode results in slightly bigger PMML documents (say, up to 10-15%), because the default split direction is encoded using extra `Node@defaultChild` and `Node@id` attributes.
The size difference disappears when the tree model is compacted.

* Added support for nullable Pandas' scalar data types.

If the dataset contains sparse columns, then they should be cast from the default Numpy `object` data type to the most appropriate nullable Pandas' scalar data type. The cast may be performed using a data type object (eg. `pandas.BooleanDtype`, `pandas.Int64Dtype`, `pandas.Float32Dtype`) or its string alias (eg. `Boolean`, `Int64`, `Float32`).

This kind of "type hinting" is instrumental to generating high(er) quality PMML documents.

## Minor improvements and fixes

* Added `ExpressionRegressor.normalization_method` attribute.

This attribute allows performing some most common normalizations atomically.

The list of supported values is `none` and `exp`.

* Refactored `ExpressionClassifier.normalization_method` attribute.

The list of supported values is `none`, `logit`, `simplemax` and `softmax`.

* Fixed the formatting of non-finite tree split values.

It is possible that some tree splits perform comparisons against the positive infinity to indicate "always true" and "always false" conditions (eg. `x <= +Inf` and `x > +Inf`, respectively).

Previously, infinite values were formatted using Java's default formatting method, which resulted in Java-style `-Infinity` and `Infinity` string literals.
They are now detected and replaced with PMML-style `-INF` and `INF` (case insensitive) string literals, respectively.

* Ensured compatibility with CHAID 5.4.1.


# 0.104.1 #

## Breaking changes

* Removed `sklearn2pmml.ensemble.OrdinalClassifier` class.

The uses of this class should be replaced with the uses of the `sklego.meta.OrdinalClassifier` class (see below), which implements exactly the same algorithm, and offers extra functionality such as calibration and parallelized fitting.

## New features

* Added support for `sklego.meta.OrdinalClassifier` class.

``` python
from pandas import CategoricalDtype, Series

# A proper ordinal target
y_bin = Series(_bin(y), dtype = CategoricalDtype(categories = [...], ordered = True), name = "bin(y)")

classifier = OrdinalClassifier(LogisticRegression(), use_calibration = True, ...)
# Map categories from objects to integer codes
classifier.fit(X, (y_bin.cat).codes.values)

# Store the categories mapping:
# the `OrdinalClassifier.classes_` attribute holds integer codes, 
# and the `OrdinalClassifier.pmml_classes_` holds the corresponding objects
classifier.pmml_classes_ = y_bin.dtype.categories
```

See [Scikit-Lego-607](https://github.com/koaning/scikit-lego/issues/607)

## Minor improvements and fixes

* Removed the SkLearn-Pandas package from installation requirements.

The `sklearn_pandas.DataFrameMapper` meta-transformer is giving way to the `sklearn.compose.ColumnTransformer` meta-transformer in most common pipelines.

* Fixed the base-N encoding of missing values.

This bug manifested itself when missing values were assigned to a category by itself.

This bug was discovered when rebuilding integration tests with Category-Encoders 2.6(.3).
It is currently unclear if the base-N encoding algorithm had its behaviour changed between Category-Encoders 2.5 and 2.6 development lines.

In any case, when using SkLearn2PMML 0.104.1 or newer, it is advisable to upgrade to Category-Encoders 2.6.0 or newer.

* Ensured compatibility with Category-Encoders 2.6.3, Imbalanced-Learn 0.12.0, OptBinning 0.19.0 and Scikit-Lego 0.7.4.


# 0.104.0 #

## Breaking changes

* Updated Scikit-Learn installation requirement from `0.18+` to `1.0+`.

This change helps the SkLearn2PMML package to better cope with breaking changes in Scikit-Learn APIs.
The underlying [JPMML-SkLearn](https://github.com/jpmml/jpmml-sklear) library retains the maximum version coverage, because it is dealing with Scikit-Learn serialized state (Pickle/Joblib or Dill), which is considerably more stable.

## New features

* Added support for Scikit-Learn 1.4.X.

The JPMML-SkLearn library integration tests were rebuilt with Scikit-Learn `1.4.0` and `1.4.1.post1` versions.
All supported transformers and estimators passed cleanly.

See [SkLearn2PMML-409](https://github.com/jpmml/sklearn2pmml/issues/409) and [JPMML-SkLearn-195](https://github.com/jpmml/jpmml-sklearn/issues/195)

* Added support for `BaseHistGradientBoosting._preprocessor` attribute.

This attribute gets initialized automatically if a `HistGradientBoostingClassifier` or `HistGradientBoostingRegressor` estimator is inputted with categorical features.

In Scikit-Learn 1.0 through 1.3 it is necessary to pre-process categorical features manually.
The indices of (ordinally-) encoded columns must be tracked and passed to the estimator using the `categorical_features` parameter:

``` python
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import OrdinalEncoder
from sklearn2pmml.decoration import CategoricalDomain, ContinuousDomain

mapper = DataFrameMapper(
  [([cont_col], ContinuousDomain()) for cont_col in cont_cols] +
  [([cat_col], [CategoricalDomain(), OrdinalEncoder()]) for cat_col in cat_cols]
)

regressor = HistGradientBoostingRegressor(categorical_features = [...])

pipeline = Pipeline([
  ("mapper", mapper),
  ("regressor", regressor)
])
pipeline.fit(X, y)
```

In Scikit-Learn 1.4, this workflow simplifies to the following:

``` python
# Activate full Pandas' support by specifying `input_df = True` and `df_out = True` 
mapper = DataFrameMapper(
  [([cont_col], ContinuousDomain()) for cont_col in cont_cols] +
  [([cat_col], CategoricalDomain(dtype = "category")) for cat_col in cat_cols]
, input_df = True, df_out = True)

# Auto-detect categorical features by their data type
regressor = HistGradientBoostingRegressor(categorical_features = "from_dtype")

pipeline = Pipeline([
  ("mapper", mapper),
  ("regressor", regressor)
])
pipeline.fit(X, y)

# Print out feature type information
# This list should contain one or more `True` values
print(pipeline._final_estimator.is_categorical_)
``` 

## Minor improvements and fixes

* Improved support for `ColumnTransformer.transformers` attribute.

Column selection using dense boolean arrays.


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

If the domain decorator needs to modify the dataset in any way (eg. performing missing or invalid value replacement), then it will create a copy of the argument dataset before modifying it.
Otherwise, the argument dataset is passed through as-is.

This aligns domain decorators with Scikit-Learn API guidelines that transformers and transformer-likes should not tamper with the original dataset.

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

Earlier SkLearn2PMML package versions may accept and convert XGBoost 2.0 without errors, but the resulting PMML document may contain an ensemble model with a wrong selection and/or wrong number of member tree models in it.
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

This brings discrete domain decorators to functional parity with continuous domain decorators, which have been supporting both single-column and multi-column mode for years.

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

* Added the `Domain` class to the `sklearn.base.OneToOneFeatureMixin` class hierarchy.

This makes domain decorators compatible with Scikit-Learn's `set_output` API.

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
