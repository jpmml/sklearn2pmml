from pycaret.internal.preprocess.transformers import TransformerWrapper
from sklearn2pmml import _escape as _default_escape

def _escape(obj, escape_func):
	obj = _default_escape(obj, escape_func = escape_func)
	if isinstance(obj, TransformerWrapper):
		obj.transformer = escape_func(obj.transformer, escape_func = escape_func)
	return obj
