from distutils.core import setup

exec(open("sklearn2pmml/metadata.py").read())

setup(
	name = "sklearn2pmml",
	version = __version__,
	description = "Python library for converting Scikit-Learn pipelines to PMML",
	author = "Villu Ruusmann",
	author_email = "villu.ruusmann@gmail.com",
	url = "https://github.com/jpmml/sklearn2pmml",
	download_url = "https://github.com/jpmml/sklearn2pmml/archive/" + __version__ + ".tar.gz",
	license = __license__,
	classifiers = [
		"Development Status :: 5 - Production/Stable",
		"Operating System :: OS Independent",
		"Programming Language :: Python",
		"Intended Audience :: Developers",
		"Intended Audience :: Science/Research",
		"Topic :: Software Development",
		"Topic :: Scientific/Engineering"
	],
	packages = [
		"sklearn2pmml",
		"sklearn2pmml.decoration",
		"sklearn2pmml.ensemble",
		"sklearn2pmml.feature_extraction",
		"sklearn2pmml.feature_extraction.text",
		"sklearn2pmml.feature_selection",
		"sklearn2pmml.pipeline",
		"sklearn2pmml.preprocessing",
		"sklearn2pmml.resources",
		"sklearn2pmml.ruleset"
	],
	package_data = {
		"sklearn2pmml.resources" : ["*.jar"]
	},
	install_requires = [
		"scikit-learn>=0.18.0",
		"sklearn-pandas>=0.0.10"
	]
)
