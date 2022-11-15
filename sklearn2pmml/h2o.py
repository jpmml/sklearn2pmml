from h2o.estimators import H2OGenericEstimator
from sklearn2pmml import EstimatorProxy

class H2OEstimatorProxy(EstimatorProxy):

	def __init__(self, estimator):
		super(H2OEstimatorProxy, self).__init__(estimator = estimator, attr_names = ["_estimator_type"])

	def download_mojo(self, path = ".", get_genmodel_jar = False, genmodel_name = ""):
		return self.estimator.download_mojo(path = path, get_genmodel_jar = get_genmodel_jar, genmodel_name = genmodel_name)

	def __getstate__(self):
		if not hasattr(self, "_mojo_path"):
			raise AttributeError("The MOJO path is not set")
		state = super(H2OEstimatorProxy, self).__getstate__()
		state["estimator"] = None
		return state

	def __setstate__(self, state):
		state["estimator"] = H2OGenericEstimator.from_file(state["_mojo_path"])
		super(H2OEstimatorProxy, self).__setstate__(state)
