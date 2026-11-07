from datetime import datetime
from numpy import datetime64
from pandas import Categorical, CategoricalDtype, DataFrame, Timestamp, Series
from sklearn.base import BaseEstimator
from sklearn2pmml.base import FeaturesInMixin, StatelessTransformerMixin

import inspect
import numpy
import pandas
import sys
import types

try:
	# Pandas 2.X
	iso8601_format = "ISO8601"

	pandas.to_datetime("2023-12-03", format = iso8601_format)
except ValueError:
	# Pandas 1.X
	iso8601_format = "%Y-%m-%dT%H:%M:%S.%f"

def _is_pandas_dataframe(X):
	return isinstance(X, DataFrame)

def _is_pandas_series(X):
	return isinstance(X, Series)

def _is_pandas_1d(X):
	return isinstance(X, (Categorical, Series))

def _is_pandas(X):
	return isinstance(X, (DataFrame, Series))

def _is_polars_dataframe(X):
	polars = sys.modules.get("polars")
	if polars is not None:
		return isinstance(X, polars.DataFrame)
	return False

def _is_polars_series(X):
	polars = sys.modules.get("polars")
	if polars is not None:
		return isinstance(X, polars.Series)
	return False

def _is_polars_1d(X):
	return _is_polars_series(X)

def _is_polars(X):
	polars = sys.modules.get("polars")
	if polars is not None:
		return isinstance(X, (polars.DataFrame, polars.Series))
	return False

def _is_series(X):
	return _is_pandas_series(X) or _is_polars_series(X)

def _is_dataframe(X):
	return _is_pandas_dataframe(X) or _is_polars_dataframe(X)

def _is_1d(X):
	return _is_pandas_1d(X) or _is_polars_1d(X)

def _has_1d_shape(X):
	shape = X.shape
	if len(shape) == 1:
		return True
	elif (len(shape) == 2 and shape[1] == 1):
		return True
	else:
		return False

def _to_1d(X):
	if _is_1d(X):
		return X
	elif _is_dataframe(X):
		columns = X.columns
		if len(columns) == 1:
			return X[columns[0]]
	X = numpy.asarray(X)
	shape = X.shape
	if len(shape) == 1:
		return X
	elif (len(shape) == 2) and (shape[1] == 1):
		return X.ravel()
	else:
		raise ValueError("Expected 1d array or 2d column vector array, got {0}d array of shape {1}".format(len(shape), str(shape)))

def _to_2d(X):
	if _is_series(X):
		return X.to_frame()
	elif len(numpy.shape(X)) == 1:
		return numpy.reshape(X, (-1, 1))
	else:
		return X

def _to_numpy(X):
	if hasattr(X, "to_numpy"):
		return X.to_numpy()
	return X

def _to_sparse(X, step_mask, step_result):
	# Make array
	if len(step_result.shape) == 1:
		result = numpy.empty((X.shape[0], ), dtype = object)
	else:
		result = numpy.empty((X.shape[0], step_result.shape[1]), dtype = object)
	# Fill array
	step_mask = numpy.asarray(step_mask).ravel()
	if len(step_result.shape) == 1:
		result[step_mask] = step_result
	else:
		result[step_mask, :] = step_result
	return result

def _copy(X):
	if _is_polars(X):
		return X.clone()
	else:
		return X.copy()

def _get_column(X, column):
	if _is_pandas_dataframe(X):
		return X[column] if isinstance(column, str) else X.iloc[:, column]
	elif _is_polars_dataframe(X):
		return X[column] if isinstance(column, str) else X.to_series(column)
	else:
		return X[:, column]

def _set_column(X, column, x):
	if _is_pandas_dataframe(X):
		X[column] = x
		return X
	elif _is_polars_dataframe(X):
		polars = sys.modules.get("polars")
		x = polars.Series(column if isinstance(column, str) else X.columns[column], x)
		return X.with_columns(x)
	else:
		X[:, column] = x
		return X

def _get_columns(X):
	if _is_dataframe(X):
		return X.columns
	else:
		return range(0, _get_column_count(X))

def _get_column_count(X):
	if hasattr(X, "shape") and len(X.shape) > 1:
		return X.shape[1]
	return 1

def _get_column_names(X):
	def _filter_column_names(X):
		return (numpy.asarray(X)).astype(str)

	if _is_pandas_series(X):
		return _filter_column_names(X.name)
	elif _is_pandas_dataframe(X):
		return _filter_column_names(X.columns.values)
	elif _is_polars_series(X):
		return _filter_column_names(X.name)
	elif _is_polars_dataframe(X):
		return _filter_column_names(X.columns)
	# elif isinstance(X, H2OFrame)
	elif hasattr(X, "names"):
		return _filter_column_names(X.names)
	else:
		return None

def _get_values(X):
	# if isinstance(X, H2OFrame)
	if hasattr(X, "as_data_frame"):
		X = X.as_data_frame()
	return _to_numpy(X)

def _set_values(X, where, values):
	if _is_polars_series(X):
		if not _is_polars_series(where):
			polars = sys.modules.get("polars")
			where = polars.Series(where)
		return X.set(where, values)
	elif _is_polars_dataframe(X):
		polars = sys.modules.get("polars")
		exprs = []
		for idx, column in enumerate(X.columns):
			column_mask = where[:, idx] if where.ndim > 1 else where
			if not column_mask.any():
				continue
			if not _is_polars_series(column_mask):
				column_mask = polars.Series(column_mask)
			exprs.append(polars.when(column_mask).then(polars.lit(values)).otherwise(polars.col(column)).alias(column))
		if exprs:
			return X.with_columns(exprs)
		return X
	else:
		X[where] = values
		return X

def _set_missing_values(X, Xt, value = None):
	if _is_polars(X):
		return Xt
	if value is None:
		value = X
	mask = pandas.isnull(X)
	if numpy.any(mask):
		if hasattr(Xt, "where"):
			Xt = Xt.where(~mask, value)
		else:
			Xt = numpy.where(~mask, Xt, value)
	return Xt

def _clear(X):
	if _is_pandas_dataframe(X):
		X.drop(index = X.index, inplace = True)
		X.drop(columns = X.columns, inplace = True)
		return X
	elif _is_polars_dataframe(X):
		return X.drop(X.columns)
	else:
		X.clear()
		return X

def _is_pandas_string(dtype):
	if hasattr(dtype, "name"):
		return dtype.name in ["str", "string"]
	return False

def _is_pandas_categorical(dtype):
	if hasattr(dtype, "name"):
		return dtype.name == "category"
	return False

def _is_pandas_proto_categorical(dtype):
	if isinstance(dtype, str) and dtype == "category":
		return True
	if isinstance(dtype, CategoricalDtype):
		return dtype.categories is None
	return False

def _is_pandas_ordinal(dtype):
	if isinstance(dtype, CategoricalDtype):
		return dtype.ordered
	return False

def _is_polars_string(dtype):
	polars = sys.modules.get("polars")
	if polars is not None:
		return dtype == polars.String
	return False

def _is_polars_categorical(dtype):
	polars = sys.modules.get("polars")
	if polars is not None:
		return isinstance(dtype, (polars.Categorical, polars.Enum))
	return False

def _is_polars_ordinal(dtype):
	polars = sys.modules.get("polars")
	if polars is not None:
		return isinstance(dtype, polars.Enum)
	return False

def _is_categorical(dtype):
	if dtype == object or dtype == str or dtype == bool:
		return True
	elif _is_pandas_string(dtype):
		return True
	elif _is_pandas_categorical(dtype):
		return True
	elif _is_polars_string(dtype):
		return True
	elif _is_polars_categorical(dtype):
		return True
	return False

def _is_ordinal(dtype):
	if _is_pandas_ordinal(dtype):
		return True
	elif _is_polars_ordinal(dtype):
		return True
	return False

def _get_categories(dtype):
	if isinstance(dtype, CategoricalDtype):
		return numpy.asarray(dtype.categories) if dtype.categories is not None else None
	polars = sys.modules.get("polars")
	if polars is not None:
		if isinstance(dtype, polars.Categorical):
			return None
		elif isinstance(dtype, polars.Enum):
			return numpy.asarray(dtype.categories)
	return None

def common_dtype(X):
	if hasattr(X, "dtype"):
		return X.dtype
	elif hasattr(X, "dtypes"):
		dtypes = set(X.dtypes)
		if len(dtypes) != 1:
			raise ValueError(dtypes)
		return next(iter(dtypes))
	else:
		raise ValueError()

def _to_numpy_dtype(dtype):
	if hasattr(dtype, "numpy_dtype"):
		return dtype.numpy_dtype
	polars = sys.modules.get("polars")
	if polars is not None:
		if isinstance(dtype, polars.DataType):
			return numpy.dtype(dtype.to_python())
	return dtype

def cast(X, dtype):
	if isinstance(dtype, str) and dtype.startswith("datetime64"):
		def eval_func(x):
			return to_pydatetime(x, dtype)

		return _eval_cells(X, eval_func, dtype = datetime)
	else:
		if _is_pandas(X):
			if isinstance(dtype, CategoricalDtype):
				if dtype.categories is not None:
					category_mask = X.isin(dtype.categories)
					X = X.where(category_mask)
			Xt = X.astype(dtype)
		elif _is_polars(X):
			Xt = X.cast(dtype)
		else:
			X = numpy.asarray(X)
			Xt = X.astype(dtype)
		if dtype in (str, "unicode"):
			Xt = _set_missing_values(X, Xt)
		return Xt

def _to_bool(X, missing_mask = None):
	if _is_pandas_1d(X):
		if missing_mask is None:
			missing_mask = pandas.isnull(X)
		if numpy.any(missing_mask):
			Xt = X.astype("boolean")
			Xt[missing_mask] = None
			return Xt
		return X.astype(bool)
	elif _is_pandas_dataframe(X):
		return X.apply(_to_bool)
	elif _is_polars(X):
		return X.cast(bool)
	else:
		if missing_mask is None:
			missing_mask = pandas.isnull(X)
		if numpy.any(missing_mask):
			Xt = X.astype(bool)
			return numpy.where(missing_mask, None, Xt)
		return X.astype(bool)

def _to_float(X, missing_mask = None):
	if _is_pandas_1d(X):
		if missing_mask is None:
			missing_mask = pandas.isnull(X)
		if numpy.any(missing_mask):
			Xt = X.astype("Float64")
			Xt[missing_mask] = None
			return Xt
		return X.astype(float)
	elif _is_pandas_dataframe(X):
		return X.apply(_to_float)
	elif _is_polars(X):
		return X.cast(float)
	else:
		if missing_mask is None:
			missing_mask = pandas.isnull(X)
		if numpy.any(missing_mask):
			Xt = X.astype(float)
			return numpy.where(missing_mask, numpy.nan, Xt)
		return X.astype(float)

def _to_int(X, missing_mask = None):
	if _is_pandas_1d(X):
		if missing_mask is None:
			missing_mask = pandas.isnull(X)
		if numpy.any(missing_mask):
			Xt = X.astype("Int64")
			Xt[missing_mask] = None
			return Xt
		return X.astype(int)
	elif _is_pandas_dataframe(X):
		return X.apply(_to_int)
	elif _is_polars(X):
		return X.cast(int)
	else:
		if missing_mask is None:
			missing_mask = pandas.isnull(X)
		if numpy.any(missing_mask):
			Xt = X.astype(float)
			return numpy.where(missing_mask, numpy.nan, Xt)
		return X.astype(int)

def _to_str(X, missing_mask = None):
	if _is_pandas_1d(X):
		if missing_mask is None:
			missing_mask = pandas.isnull(X)
		if numpy.any(missing_mask):
			Xt = X.astype("string")
			Xt[missing_mask] = None
			return Xt
		return X.astype(str)
	elif _is_pandas_dataframe(X):
		return X.apply(_to_str)
	elif _is_polars(X):
		return X.cast(str)
	else:
		if missing_mask is None:
			missing_mask = pandas.isnull(X)
		if numpy.any(missing_mask):
			Xt = X.astype(str)
			return numpy.where(missing_mask, None, Xt)
		return X.astype(str)

def ensure_def(expr, env):
	expr_code = compile(expr, "<string>", "exec")

	name = expr_code.co_names[0]

	if name in globals():
		return globals()[name]

	try:
		return env[name]
	except KeyError:
		eval(expr_code, env)

		return env[name]

class Evaluatable:

	def __init__(self, expr, function_defs = []):
		if isinstance(expr, str):
			if "\n" in expr:
				raise ValueError()
			self.expr = expr
		else:
			raise TypeError()

		def to_source(function_def):
			if isinstance(function_def, str):
				if "\n" not in function_def:
					raise ValueError()
				if "def" not in function_def:
					raise ValueError()
				return function_def
			elif isinstance(function_def, types.FunctionType):
				return inspect.getsource(function_def)
			else:
				raise TypeError()

		self.function_defs = [to_source(function_def) for function_def in function_defs]

	def uses(self, module):
		if (module + ".") in self.expr:
			return True
		for function_def in self.function_defs:
			if (module + ".") in function_def:
				return True
		return False

	def setup(self, env):
		for function_def in self.function_defs:
			ensure_def(function_def, env)
		main_function_def = "def _evaluate(X):\n\treturn ({})".format(self.expr)
		ensure_def(main_function_def, env)

	def evaluate(self, X, env):
		func = env["_evaluate"]
		return func(X)

class Expression(Evaluatable):

	def __init__(self, expr, function_defs = []):
		super(Expression, self).__init__(expr = expr, function_defs = function_defs)

class Predicate(Evaluatable):

	def __init__(self, expr, function_defs = []):
		super(Predicate, self).__init__(expr = expr, function_defs = function_defs)

def check_expression(expression):
	if not isinstance(expression, (str, Expression)):
		raise TypeError("The expression object is not a string nor an instance of {0}".format(Expression.__name__))
	return expression

def check_predicate(predicate):
	if not isinstance(predicate, (str, Predicate)):
		raise TypeError("The predicate object is not a string nor an instance of {0}".format(Predicate.__name__))
	return predicate

def to_expr(expr):
	if isinstance(expr, str):
		return expr
	elif isinstance(expr, types.FunctionType):
		if expr.__code__.co_argcount != 1:
			raise ValueError()
		if expr.__code__.co_varnames[0] != "X":
			raise ValueError()
		return inspect.getsource(expr)
	elif isinstance(expr, Evaluatable):
		return expr
	else:
		raise TypeError()

def to_evaluatable_func(expr, modules = ["math", "re", "pcre", "pcre2", "numpy", "pandas", "scipy"]):
	env = dict()

	if isinstance(expr, str):
		for module in modules:
			if (module + ".") in expr:
				exec("import {}".format(module), env)

		if "\n" not in expr:
			expr_code = compile(expr, "<string>", "eval")

			def evaluate(x):
				env["X"] = x
				return eval(expr_code, env)

			return evaluate
		else:
			func = ensure_def(expr, env)
			return lambda x: func(x)
	elif isinstance(expr, Evaluatable):
		for module in modules:
			if expr.uses(module):
				exec("import {}".format(module), env)

		expr.setup(env = env)
		return lambda x: expr.evaluate(x, env)
	else:
		raise TypeError()

class _SeriesApplier:

	def __init__(self, X):
		if not _is_series(X):
			raise TypeError()
		self.X = X

	def apply(self, func):
		X = self.X
		if _is_pandas_series(X):
			return X.apply(func)
		elif _is_polars_series(X):
			return X.map_elements(func, skip_nulls = False)
		else:
			raise TypeError()

class _DataFrameApplier:

	def __init__(self, X):
		if not _is_dataframe(X):
			raise TypeError()
		self.X = X
		self.column_indexes = {column: index for index, column in enumerate(X.columns)}
		self._values = None

	def __array__(self, dtype = None):
		return numpy.asarray(self._values, dtype = dtype)

	def __getitem__(self, key):
		if isinstance(key, str):
			key = self.column_indexes[key]
		return self._values[key]

	def __iter__(self):
		return iter(self._values)

	def apply(self, func):
		X = self.X
		nrow = X.shape[0]
		Xt = numpy.empty(shape = (nrow, ), dtype = object)
		if _is_pandas_dataframe(X):
			rows = X.itertuples(index = False, name = None)
		elif _is_polars_dataframe(X):
			rows = X.iter_rows(named = False)
		else:
			raise TypeError()
		for index, values in enumerate(rows):
			self._values = values
			Xt[index] = func(self)
		self._values = None
		if _is_pandas_dataframe(X):
			return Series(Xt, index = X.index)
		elif _is_polars_dataframe(X):
			polars = sys.modules.get("polars")
			return polars.Series(Xt)
		else:
			raise TypeError()

class _NDArrayApplier:

	def __init__(self, X):
		self.X = X

	def apply(self, func):
		X = self.X
		nrow = X.shape[0]
		Xt = numpy.empty(shape = (nrow, ), dtype = object)
		for i in range(0, nrow):
			Xt[i] = func(X[i])
		return Xt

def _eval_array_rows(X, func, to_numpy = False, shape = None, dtype = None):
	X = _to_2d(X)
	if _is_dataframe(X):
		Xt = _DataFrameApplier(X).apply(func)
		if dtype is not None:
			Xt = cast(Xt, dtype)
		if to_numpy:
			Xt = _to_numpy(Xt)
			if shape is not None:
				Xt = Xt.reshape(shape)
		return Xt
	else:
		return _eval_ndarray_rows(X, func, shape = shape, dtype = dtype)

def _eval_scalar_rows(X, func, shape = None, dtype = None):
	if _is_series(X):
		Xt = _SeriesApplier(X).apply(func)
		if dtype is not None:
			Xt = cast(Xt, dtype)
		return Xt
	else:
		return _eval_ndarray_rows(X, func, shape = shape, dtype = dtype)

def _eval_ndarray_rows(X, func, shape = None, dtype = None):
	Xt = _NDArrayApplier(X).apply(func)
	if dtype is not None:
		Xt = cast(Xt, dtype)
	if shape is not None:
		Xt = Xt.reshape(shape)
	return Xt

def _eval_cells(X, func, dtype = None):
	if _is_pandas_1d(X):
		return X.apply(func)
	elif _is_pandas_dataframe(X):
		# Pandas 2.0.X
		if not hasattr(X, "map"):
			return X.applymap(func)
		# Pandas 2.1.0+
		else:
			return X.map(func)
	elif _is_polars_series(X):
		return X.map_elements(func, return_dtype = dtype)
	elif _is_polars_dataframe(X):
		polars = sys.modules.get("polars")
		return X.select(polars.all().map_elements(func, return_dtype = dtype))
	else:
		X = numpy.asarray(X)
		Xt = numpy.vectorize(func, otypes = [object])(X)
		return Xt

def to_pydatetime(obj, dtype):
	if pandas.isnull(obj):
		return None
	if not isinstance(obj, (str, datetime, datetime64, Timestamp)):
		raise TypeError()
	ts = pandas.to_datetime(obj, yearfirst = True, format = iso8601_format, origin = "unix")
	if dtype == "datetime64[D]":
		ts = ts.floor(freq = "D")
	elif dtype == "datetime64[s]":
		ts = ts.floor(freq = "s")
	else:
		raise ValueError(dtype)
	return ts.to_pydatetime()

def fqn(obj):
	clazz = obj if inspect.isclass(obj) else obj.__class__
	return ".".join([clazz.__module__, clazz.__name__])

def is_instance_attr(obj, name):
	if not hasattr(obj, name):
		return False
	if name.startswith("__") and name.endswith("__"):
		return False
	v = getattr(obj, name)
	if isinstance(v, (types.BuiltinFunctionType, types.BuiltinMethodType, types.FunctionType, types.MethodType)):
		return False
	# See https://stackoverflow.com/a/17735709/
	attr_type = getattr(type(obj), name, None)
	if isinstance(attr_type, property):
		return False
	return True

def get_instance_attrs(obj):
	names = dir(obj)
	names = [name for name in names if is_instance_attr(obj, name)]
	return names

def sizeof(obj, with_overhead = False):
	if with_overhead:
		return sys.getsizeof(obj)
	return obj.__sizeof__()

def deep_sizeof(obj, with_overhead = False, verbose = False):
	# Primitive type values
	if obj is None:
		return obj.__sizeof__()
	elif isinstance(obj, (int, float, str, bool, numpy.int64, numpy.float32, numpy.float64)):
		return obj.__sizeof__()
	# Iterables
	elif isinstance(obj, list):
		sum = sizeof([], with_overhead = with_overhead) # Empty list
		for v in obj:
			v_sizeof = deep_sizeof(v, with_overhead = with_overhead, verbose = False)
			sum += v_sizeof
		return sum
	elif isinstance(obj, tuple):
		sum = sizeof((), with_overhead = with_overhead) # Empty tuple
		for i, v in enumerate(obj):
			v_sizeof = deep_sizeof(v, with_overhead = with_overhead, verbose = False)
			sum += v_sizeof
		return sum
	# Numpy ndarrays
	elif isinstance(obj, numpy.ndarray):
		sum = sizeof(obj, with_overhead = with_overhead) # Array header
		sum += (obj.size * obj.itemsize) # Array content
		return sum
	# Reference type values
	else:
		qualname = fqn(obj)
		# Restrict the circle of competence to Scikit-Learn classes
		if not (qualname.startswith("_abc.") or qualname.startswith("sklearn.")):
			raise TypeError("The object (class {0}) is not supported ".format(qualname))
		sum = sizeof(object(), with_overhead = with_overhead) # Empty object
		names = get_instance_attrs(obj)
		if names:
			if verbose:
				print("| Attribute | `type(v)` | `deep_sizeof(v)` |")
				print("|---|---|---|")
			for name in names:
				v = getattr(obj, name)
				v_type = type(v)
				v_sizeof = deep_sizeof(v, with_overhead = with_overhead, verbose = False)
				sum += v_sizeof
				if verbose:
					print("| {} | {} | {} |".format(name, v_type, v_sizeof))
		return sum

class Reshaper(BaseEstimator, FeaturesInMixin, StatelessTransformerMixin):

	def __init__(self, newshape):
		self.newshape = newshape

	def fit(self, X, y = None):
		self._check_features_in(X, reset = True)
		return self

	def transform(self, X):
		return X.reshape(self.newshape)

class Slicer(BaseEstimator, FeaturesInMixin, StatelessTransformerMixin):

	def __init__(self, start = None, stop = None, step = None):
		self.start = start
		self.stop = stop
		if (step is not None) and (step <= 0):
			raise ValueError("Step must be positive integer")
		self.step = step

	def fit(self, X, y = None):
		self._check_features_in(X, reset = True)
		return self

	def transform(self, X):
		rows = slice(None, None)
		columns = slice(self.start, self.stop, self.step)
		if _is_pandas_dataframe(X):
			return X.iloc[rows, columns]
		else:
			return X[rows, columns]
