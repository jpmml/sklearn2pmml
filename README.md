SkLearn2PMML
============

Python library for converting [Scikit-Learn] (http://scikit-learn.org/) models to PMML.

# Features #

This library is a thin wrapper around the [JPMML-SkLearn] (https://github.com/jpmml/jpmml-sklearn) command-line application. For a list of supported Scikit-Learn Estimator and Transformer types, please refer to the documentation of the JPMML-SkLearn project.

# Prerequisites #

* Python 2.7, 3.4 or newer.
* Java 1.7 or newer. The Java executable must be available on system path.

# Installation #

Installing the latest version from GitHub:

```
pip install --user --upgrade git+https://github.com/jpmml/sklearn2pmml.git
```

# Usage #

A typical workflow can be summarized as follows:

1. Create and fit a [`sklearn_pandas.DataFrameMapper`] (https://pypi.python.org/pypi/sklearn-pandas) object that captures "feature engineering" operations that are needed to transform data from its original representation to the Scikit-Learn's representation.
2. Create and fit a Scikit-Learn's Estimator object.
3. Convert the above two Python objects to a PMML document by invoking the utility method `sklearn2pmml.sklearn2pmml(estimator, mapper, pmml_destination_path)`.

For example, developing a logistic regression model for the classification of iris species:

```python
#
# Step 1: feature engineering
#

from sklearn.decomposition import PCA
from sklearn_pandas import DataFrameMapper

import pandas

iris_df = pandas.read_csv("Iris.csv")

iris_mapper = DataFrameMapper([
    (["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"], PCA(n_components = 3)),
    ("Species", None)
])

iris_df = iris_mapper.fit_transform(iris_df)

#
# Step 2: training a logistic regression model
#

from sklearn.linear_model import LogisticRegressionCV

iris_X = iris_df[:, 0:3]
iris_y = iris_df[:, 3]

iris_estimator = LogisticRegressionCV()
iris_estimator.fit(iris_X, iris_y)

#
# Step 3: conversion to PMML
#

import sklearn2pmml

sklearn2pmml.sklearn2pmml(iris_estimator, iris_mapper, "LogisticRegressionIris.pmml")
```

# De-installation #

Uninstalling:

```
pip uninstall sklearn2pmml
```

# License #

SkLearn2PMML is dual-licensed under the [GNU Affero General Public License (AGPL) version 3.0] (http://www.gnu.org/licenses/agpl-3.0.html) and a commercial license.

# Additional information #

Please contact [info@openscoring.io] (mailto:info@openscoring.io)