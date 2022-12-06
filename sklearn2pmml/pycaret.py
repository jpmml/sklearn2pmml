from pycaret.internal.preprocess.transformers import TransformerWrapper
from sklearn2pmml import _escape as _default_escape
from sklearn2pmml import make_pmml_pipeline as default_make_pmml_pipeline

def _escape(obj, escape_func):
	obj = _default_escape(obj, escape_func = escape_func)
	if isinstance(obj, TransformerWrapper):
		obj.transformer = escape_func(obj.transformer, escape_func = escape_func)
	return obj

def make_pmml_pipeline(obj, active_fields = None, target_fields = None, escape_func = _escape):
	return default_make_pmml_pipeline(obj, active_fields = active_fields, target_fields = target_fields, escape_func = escape_func)
