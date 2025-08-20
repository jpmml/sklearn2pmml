from argparse import ArgumentParser, BooleanOptionalAction
from sklearn2pmml import __version__, sklearn2pmml

import dill
import joblib

def main():
	version = "SkLearn2PMML {}".format(__version__)

	parser = ArgumentParser(prog = "sklearn2pmml", description = "SkLearn2PMML command-line application")
	parser.add_argument("-i", "--input", type = str, required = True, help = "Input Pickle file")
	parser.add_argument("-o", "--output", type = str, required = True, help = "Output PMML file")
	parser.add_argument("--unpickle", action = BooleanOptionalAction, default = True, help = "Unpickle the pickle file to make its content modifiable")
	parser.add_argument("--schema", type = str, help = "Output PMML schema version")
	parser.add_argument("--version", action = "version", version = version)

	args = parser.parse_args()

	if args.unpickle:
		try:
			args.input = joblib.load(args.input)
		except Exception as joblib_error:
			try:
				with open(args.input, "rb") as dill_file:
					args.input = dill.load(dill_file)
			except Exception as dill_error:
				raise joblib_error

	sklearn2pmml(args.input, pmml_path = args.output, pmml_schema = args.schema)
