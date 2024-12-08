from setuptools import find_packages, setup

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
	packages = find_packages(exclude = ["*.tests"]),
	package_data = {
		"sklearn2pmml.resources" : ["classpath.txt", "*.jar"]
	},
	exclude_package_data = {
		"" : ["README.md"],
	},
	entry_points={
		"console_scripts" : [
			"sklearn2pmml=sklearn2pmml.cli:main",
		],
	},
	python_requires = ">=3.8",
	install_requires = [
		"dill>=0.3.4",
		"joblib>=0.13.0",
		"pandas>=1.5.0",
		"scikit-learn>=1.0"
	]
)
