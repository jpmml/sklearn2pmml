from sklearn2pmml.preprocessing import ExpressionTransformer

class BusinessDecisionTransformer(ExpressionTransformer):

	def __init__(self, expr, business_problem, decisions, dtype = None):
		super(BusinessDecisionTransformer, self).__init__(expr = expr, dtype = dtype)
		self.business_problem = business_problem
		for decision in decisions:
			if type(decision) is not tuple:
				raise ValueError("Decision is not a tuple")
			if len(decision) != 2:
				raise ValueError("Decision is not a two-element (value, description) tuple")
		self.decisions = decisions
