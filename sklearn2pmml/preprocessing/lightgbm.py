from sklearn_pandas import DataFrameMapper
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn2pmml import _is_categorical
from sklearn2pmml.preprocessing import PMMLLabelEncoder

def make_lightgbm_dataframe_mapper(dtypes, missing_value_aware = True):
	"""Construct a DataFrameMapper for feeding complex data into a LGBMModel.

	Parameters
	----------

	dtypes: iterable of tuples (column, dtype)

	missing_value_aware: boolean
		If true, use missing value aware transformers.

	Returns
	-------
	Tuple (DataFrameMapper, list of categorical columns indices)

	"""
	features = list()
	categorical_features = list()
	i = 0
	for column, dtype in dtypes.items():
		if _is_categorical(dtype):
			features.append(([column], PMMLLabelEncoder(missing_values = -1) if missing_value_aware else LabelEncoder()))
			categorical_features.append(i)
		else:
			features.append(([column], None))
		i += 1
	return (DataFrameMapper(features), categorical_features)

def make_lightgbm_column_transformer(dtypes, missing_value_aware = True):
	"""Construct a ColumnTransformer for feeding complex data into a LGBMModel.

	Parameters
	----------

	dtypes: iterable of tuples (column, dtype)

	missing_value_aware: boolean
		If true, use missing value aware transformers.

	Returns:
	Tuple (ColumnTransformer, list of categorical column indices)

	"""
	transformers = list()
	categorical_features = list()
	i = 0
	for column, dtype in dtypes.items():
		if _is_categorical(dtype):
			transformers.append((str(column), PMMLLabelEncoder(missing_values = -1) if missing_value_aware else OrdinalEncoder(), [column]))
			categorical_features.append(i)
		else:
			transformers.append((str(column), "passthrough", [column]))
		i += 1
	return (ColumnTransformer(transformers, remainder = "drop"), categorical_features)