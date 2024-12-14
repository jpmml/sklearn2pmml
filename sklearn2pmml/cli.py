from argparse import ArgumentParser
from sklearn2pmml import __version__, sklearn2pmml

def main():
	version = "SkLearn2PMML {}".format(__version__)

	parser = ArgumentParser(prog = "sklearn2pmml", description = "SkLearn2PMML command-line application")
	parser.add_argument("-i", "--input", type = str, required = True, help = "Input Pickle file")
	parser.add_argument("-o", "--output", type = str, required = True, help = "Output PMML file")
	parser.add_argument("--schema", type = str, help = "Output PMML schema version")
	parser.add_argument("--version", action = "version", version = version)

	args = parser.parse_args()

	sklearn2pmml(args.input, pmml_path = args.output, pmml_schema = args.schema)
