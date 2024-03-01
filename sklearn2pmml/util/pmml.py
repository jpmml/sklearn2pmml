from lxml import etree
from pandas.api.types import is_string_dtype, is_integer_dtype

NS_PMML44 = "{http://www.dmg.org/PMML-4_4}"

def make_element(tag, **_extra):
	return etree.Element(NS_PMML44 + tag, **_extra)

class PMMLElement(object):

	def __init__(self, root):
		self.root = root

	def tostring(self, encoding = "utf-8"):
		return etree.tostring(self.root, pretty_print = True).decode(encoding)

	def set(self, name, value):
		self.root.set(name, str(value))
		return self

	def append(self, element):
		self.root.append(element.root if isinstance(element, PMMLElement) else element)
		return self

	def text(self, text):
		self.root.text = text

class Array(PMMLElement):

	def __init__(self, values):
		super(Array, self).__init__(make_element("Array"))

		if is_string_dtype(values):
			vtype = "string"
		elif is_integer_dtype(values):
			vtype = "int"
		else:
			vtype = "real"

		self.set("n", str(len(values)))
		self.set("type", vtype)
		self.text(" ".join([str(value) for value in values]))

class Extension(PMMLElement):

	def __init__(self, name = None, value = None):
		super(Extension, self).__init__(make_element("Extension"))

		if name is not None:
			self.set("name", name)
		if value is not None:
			self.set("value", value)
