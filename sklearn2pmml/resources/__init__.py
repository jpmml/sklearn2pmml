from importlib.resources import path

def _package_classpath():
	jars = []
	with path("sklearn2pmml.resources", "classpath.txt") as classpath_file:
		resources_dir = classpath_file.parent
		with open(str(classpath_file), "r") as classpath:
			jar_names = [line.strip("\n") for line in classpath.readlines()]
			for jar_name in jar_names:
				jar_file = resources_dir / jar_name
				jars.append(str(jar_file))
	return jars
