def add_options(estimator, **options):
	if hasattr(estimator, "pmml_options_"):
		pmml_options = dict(estimator.pmml_options_)
		pmml_options.update(options)
	else:
		pmml_options = dict(options)
	estimator.pmml_options_ = pmml_options

def clear_options(estimator):
	if hasattr(estimator, "pmml_options_"):
		delattr(estimator, "pmml_options_")
