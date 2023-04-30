from sklearn_pandas import DataFrameMapper
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, OrdinalEncoder
from sklearn2pmml import _is_categorical
from sklearn2pmml.preprocessing import PMMLLabelBinarizer

def make_xgboost_dataframe_mapper(dtypes, missing_value_aware = True):
	"""Construct a DataFrameMapper for feeding complex data into an XGBModel.

	Parameters:
	----------

	dtypes: iterable of tuples (column, dtype)

	missing_value_aware: boolean
		If true, use missing value aware transformers.

	Returns:
	-------
	DataFrameMapper

	"""
	features = list()
	for column, dtype in dtypes.items():
		if _is_categorical(dtype):
			features.append(([column], PMMLLabelBinarizer(sparse_output = True) if missing_value_aware else LabelBinarizer(sparse_output = True)))
		else:
			features.append(([column], None))
	return DataFrameMapper(features)

def make_xgboost_column_transformer(dtypes, missing_value_aware = True):
	"""Construct a ColumnTransformer for feeding complex data into an XGBModel.

	Parameters:
	----------

	dtypes: iterable of tuples (column, dtype)

	missing_value_aware: boolean
		If true, use missing value aware transformers.

	Returns:
	-------
	ColumnTransformer

	"""
	transformers = list()
	for column, dtype in dtypes.items():
		if _is_categorical(dtype):
			transformers.append((str(column), PMMLLabelBinarizer(sparse_output = True) if missing_value_aware else Pipeline([("ordinal_encoder", OrdinalEncoder()), ("one_hot_encoder", OneHotEncoder())]), [column]))
		else:
			transformers.append((str(column), "passthrough", [column]))
	return ColumnTransformer(transformers, remainder = "drop")
