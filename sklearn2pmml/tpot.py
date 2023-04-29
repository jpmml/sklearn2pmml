from sklearn2pmml import _strip_module, load_class_mapping

def make_pmml_config(config, user_classpath = []):
	"""Translates a regular TPOT configuration to a PMML-compatible TPOT configuration.

	Parameters:
	----------
	obj: config
		The configuration dictionary.

	user_classpath: list of strings, optional
		The paths to JAR files that provide custom Transformer, Selector and/or Estimator converter classes.
		The SkLearn2PMML classpath is constructed by appending user JAR files to package JAR files.

	"""
	keys = set(config.keys())
	mapping = load_class_mapping(user_classpath)
	classes = mapping.keys()
	pmml_keys = (set(classes)).union(set([_strip_module(class_) for class_ in classes]))
	return { key : config[key] for key in (keys).intersection(pmml_keys)}