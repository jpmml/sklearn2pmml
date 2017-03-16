from distutils.core import setup
import sys
sys.path.append('sklearn2pmml')
from version import __version__
from license import __license__

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
		"sklearn2pmml.feature_extraction",
		"sklearn2pmml.feature_extraction.text",
		"sklearn2pmml.resources"
	],
	package_data = {
		"sklearn2pmml.resources" : ["*.jar"]
	},
	install_requires = [
		"scikit-learn>=0.16.0",
		"sklearn_pandas>=0.0.10"
	]
)
