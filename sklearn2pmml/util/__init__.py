from pandas import Categorical, DataFrame, Index, Series
from sklearn.base import clone, BaseEstimator, TransformerMixin

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

def cast(X, dtype):
	if isinstance(dtype, str) and dtype.startswith("datetime64"):
		func = lambda x: to_pydatetime(x, dtype)
		return dt_transform(X, func)
	else:
		if not hasattr(X, "astype"):
			X = numpy.asarray(X)
		return X.astype(dtype)

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

def to_numpy(X):
	if hasattr(X, "to_numpy"):
		return X.to_numpy()
	return X

def is_1d(X):
	shape = X.shape
	if len(shape) == 1:
		return True
	elif (len(shape) == 2 and shape[1] == 1):
		return True
	else:
		return False

def to_1d(X):
	if isinstance(X, (Categorical, Series)):
		return X
	elif isinstance(X, DataFrame):
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

def dt_transform(X, func):
	if hasattr(X, "applymap"):
		return X.applymap(func)
	shape = X.shape
	if len(shape) > 1:
		X = X.ravel()
	Xt = func(X)
	Xt = to_numpy(Xt)
	if len(shape) > 1:
		Xt = Xt.reshape(shape)
	return Xt

def to_pydatetime(X, dtype):
	Xt = pandas.to_datetime(X, yearfirst = True, format = iso8601_format, origin = "unix")
	if hasattr(Xt, "dt"):
		Xt = Xt.dt
	if dtype == "datetime64[D]":
		Xt = Xt.floor(freq = "D")
	elif dtype == "datetime64[s]":
		Xt = Xt.floor(freq = "s")
	else:
		raise ValueError(dtype)
	return Xt.to_pydatetime()

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

def to_expr_func(expr, modules = ["math", "re", "pcre", "pcre2", "numpy", "pandas", "scipy"]):
	env = dict()

	if isinstance(expr, str):
		for module in modules:
			if (module + ".") in expr:
				exec("import {}".format(module), env)

		if "\n" not in expr:

			def evaluate(x):
				env["X"] = x
				return eval(expr, env)

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

def eval_rows(X, func, to_numpy = False, shape = None, dtype = None):
	if hasattr(X, "apply"):
		if isinstance(X, Series):
			Xt = X.apply(func)
		else:
			Xt = X.apply(func, axis = 1)
		if dtype is not None:
			Xt = Xt.astype(dtype)
		if to_numpy:
			Xt = Xt.to_numpy()
			if shape is not None:
				Xt = Xt.reshape(shape)
	else:
		nrow = X.shape[0]
		Xt = numpy.empty(shape = (nrow, ), dtype = (dtype if dtype is not None else object))
		for i in range(0, nrow):
			Xt[i] = func(X[i])
		if shape is not None:
			Xt = Xt.reshape(shape)
	return Xt

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

class Reshaper(BaseEstimator, TransformerMixin):

	def __init__(self, newshape):
		self.newshape = newshape

	def fit(self, X, y = None):
		return self

	def transform(self, X):
		return X.reshape(self.newshape)

class Slicer(BaseEstimator, TransformerMixin):

	def __init__(self, start = None, stop = None, step = None):
		self.start = start
		self.stop = stop
		if (step is not None) and (step <= 0):
			raise ValueError("Step must be positive integer")
		self.step = step

	def fit(self, X, y = None):
		return self

	def transform(self, X):
		rows = slice(None, None)
		columns = slice(self.start, self.stop, self.step)
		if isinstance(X, DataFrame):
			return X.iloc[rows, columns]
		else:
			return X[rows, columns]
