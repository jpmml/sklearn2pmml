from distutils.core import setup

from sklearn2pmml import __license__, __version__

setup(
	name = "sklearn2pmml",
	version = __version__,
	description = "Python library for converting Scikit-Learn models to PMML",
	author = "Villu Ruusmann",
	author_email = "villu.ruusmann@gmail.com",
	url = "https://github.com/jpmml/sklearn2pmml",
	license = __license__,
	packages = [
		"sklearn2pmml",
		"sklearn2pmml.decoration",
		"sklearn2pmml.resources"
	],
	package_data = {
		"sklearn2pmml.resources" : ["*.jar"]
	},
	install_requires = [
		"joblib",
		"numpy",
		"sklearn",
		"sklearn_pandas"
	]
)
