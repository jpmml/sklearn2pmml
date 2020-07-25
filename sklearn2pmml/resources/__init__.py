import pkg_resources

def _package_classpath():
	jars = []
	with open(pkg_resources.resource_filename("sklearn2pmml.resources", "classpath.txt")) as classpath:
		jars = [pkg_resources.resource_filename("sklearn2pmml.resources", jar_name.strip()) for jar_name in classpath.readlines()]
	return jars