# 0.101.0

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

If the `CastTransformer.dtype` parameter value is "category" (ie. string literal), then the fit method will auto-detect valid category levels, and will set the `CastTransformer.dtype_` attribute to a `pandas.CategoricalDtype` object instead.
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
