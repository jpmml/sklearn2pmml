import numpy

def add_customizations(estimator, customizations):
	for customization in customizations:
		if not isinstance(customization, Customization):
			raise TypeError()
	if hasattr(estimator, "pmml_customizations_"):
		pmml_customizations = estimator.pmml_customizations_
		pmml_customizations = numpy.append(pmml_customizations, customizations)
	else:
		pmml_customizations = numpy.asarray(customizations)
	estimator.pmml_customizations_ = pmml_customizations

def clear_customizations(estimator):
	if hasattr(estimator, "pmml_customizations_"):
		delattr(estimator, "pmml_customizations_")

class Customization(object):

	def __init__(self, command, xpath_expr, pmml_element):
		commands = ["insert", "update", "delete"]
		if command not in commands:
			raise ValueError("Command {} not in {}".format(command, commands))
		self.command = command
		if xpath_expr is not None:
			if not isinstance(xpath_expr, str):
				raise TypeError("XPath expression is not a string")
		else:
			if command in ["delete"]:
				raise ValueError("Command {} requires XPath expression".format(command))
		self.xpath_expr = xpath_expr
		# XXX: isinstance(pmml, sklearn2pmml.metrics.PMMLElement)
		if hasattr(pmml_element, "tostring"):
			pmml_element = pmml_element.tostring()
		if pmml_element is not None:
			if not isinstance(pmml_element, str):
				raise TypeError("PMML element is not a string")
			if command in ["delete"]:
				raise ValueError("Command {} does not support PMML element".format(command))
		else:
			if command in ["insert", "update"]:
				raise ValueError("Command {} requires PMML element".format(command))
		self.pmml_element = pmml_element
