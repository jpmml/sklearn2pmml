from pandas import BooleanDtype, DataFrame, Float64Dtype, Int64Dtype
from sklearn2pmml.xgboost import make_feature_map
from sklearn2pmml.xgboost import FeatureMap
from unittest import TestCase

class FunctionTest(TestCase):

	def test_make_feature_map(self):
		df = DataFrame([
			["orange", "orange", 150, False, 4.5],
			["apple", "red", 150, True, 4.0],
			["banana", "yellow", 100, True, 3.5],
			["apple", "green", 200, False, 5]
		], columns = ["fruit", "color", "weight", "organic", "rating"])
		df["fruit"] = df["fruit"].astype("category")
		df["color"] = df["color"].astype("category")
		df["weight"] = df["weight"].astype(int)
		df["organic"] = df["organic"].astype(bool)
		df["rating"] = df["rating"].astype(float)
		with self.assertRaises(ValueError):
			make_feature_map(df, enable_categorical = False)
		fmap = make_feature_map(df, enable_categorical = True)
		self.assertIsInstance(fmap, FeatureMap)
		self.assertIsInstance(fmap, DataFrame)
		self.assertEqual((10, 3), fmap.shape)
		self.assertEqual(["id", "name", "type"], fmap.columns.tolist())
		self.assertEqual([0, "fruit=apple", "i"], fmap.iloc[0].tolist())
		self.assertEqual([3, "color=green", "i"], fmap.iloc[3].tolist())
		self.assertEqual([7, "weight", "int"], fmap.iloc[7].tolist())
		self.assertEqual([8, "organic", "i"], fmap.iloc[8].tolist())
		self.assertEqual([9, "rating", "q"], fmap.iloc[9].tolist())
		df_nullable = df.copy()
		df_nullable["weight"] = df_nullable["weight"].astype(Int64Dtype())
		df_nullable["organic"] = df_nullable["organic"].astype(BooleanDtype())
		df_nullable["rating"] = df_nullable["rating"].astype(Float64Dtype())
		fmap_nullable = make_feature_map(df, enable_categorical = True)
		self.assertEqual(fmap.shape, fmap_nullable.shape)
		self.assertEqual(fmap.values.tolist(), fmap_nullable.values.tolist())
