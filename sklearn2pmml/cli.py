from argparse import ArgumentParser
from sklearn2pmml import sklearn2pmml

import joblib

def main():
	parser = ArgumentParser(prog = "sklearn2pmml", description = "SkLearn2PMML command-line application")
	parser.add_argument("-i", "--input", type = str, required = True, help = "Input Pickle file")
	parser.add_argument("-o", "--output", type = str, required = True, help = "Output PMML file")
	parser.add_argument("--schema", type = str, help = "Output PMML schema version")

	args = parser.parse_args()

	estimator = joblib.load(args.input)

	sklearn2pmml(estimator, pmml_path = args.output, pmml_schema = args.schema)
