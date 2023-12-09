from pandas import DataFrame

class FeatureMap(DataFrame):

	def __init__(self, data):
		super(FeatureMap, self).__init__(data = data, columns = ["id", "name", "type"])

	def save(self, path):
		self.to_csv(path, sep = "\t", header = False, index = False)

def make_feature_map(df, enable_categorical = True, category_to_indicator = True):
	entries = []

	feature_names = df.columns.tolist()

	for feature_name in feature_names:
		dtype = df[feature_name].dtype

		if dtype == "category" and enable_categorical:
			if category_to_indicator:
				categories = dtype.categories
				for category in categories:
					entries.append(("{}={}".format(feature_name, category), "i"))
			else:
				entries.append((feature_name, "c"))
		elif dtype in ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "Int8", "Int16", "Int32", "Int64"):
			entries.append((feature_name, "int"))
		elif dtype in ("float16", "float32", "float64", "Float32", "Float64"):
			entries.append((feature_name, "q"))
		elif dtype in ("bool", "boolean"):
			entries.append((feature_name, "i"))
		else:
			raise ValueError(dtype)

	entries = [(idx, entry[0], entry[1]) for idx, entry in enumerate(entries)]

	return FeatureMap(entries)