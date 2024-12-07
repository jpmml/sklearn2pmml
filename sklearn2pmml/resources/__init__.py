from pathlib import Path

def _package_classpath():
	sklearn2pmml_dir = Path(__import__("sklearn2pmml").__file__).parent
	resources_dir = sklearn2pmml_dir / "resources"

	jars = []
	classpath_file = resources_dir / "classpath.txt"
	with open(str(classpath_file), "r") as classpath:
		jar_names = [line.rstrip("\n") for line in classpath.readlines()]
		for jar_name in jar_names:
			jar_file = resources_dir / jar_name
			jars.append(str(jar_file))
	return jars
