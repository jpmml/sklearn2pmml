import sklearn

# SkLearn 1.7.0+
def _ensure_tosequence():
	try:
		from sklearn.utils import tosequence
	except ImportError:
		def tosequence(x):
			if isinstance(x, (list, tuple)):
				return x
			return list(x) if hasattr(x, '__iter__') else [x]

		sklearn.utils.tosequence = tosequence

def patch_sklearn():
	_ensure_tosequence()
