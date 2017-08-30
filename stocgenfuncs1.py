from tfsource import *
from datasource import *
from paramfunctions1 import *



def simple_model_1(cat_dist, cont_var_genfuncs, prob_funcs):
	catvals = len(cat_dist)
	def sample():
		catval = np.random.choice(catvals, p = cat_dist)
		contvarval = cont_var_genfuncs[catval]()
		prob = prob_funcs[catval](contvarval)
		outcome = np.random.binomial(1, prob)
		return [catval, contvarval, prob, outcome]
	return sample

simpmod1_ex1 = \
simple_model_1([0.2,0.4,0.4], \
	[lambda : np.random.uniform(0, 100), lambda : np.random.uniform(40, 120), lambda : np.random.uniform(-20, 80)],\
	[lambda x : 0.1, lambda x : 0.5, lambda x : 0.9])


simpmod1_ex2 = \
simple_model_1([0.45,0.1,0.45], \
	[lambda : np.random.uniform(0, 100), \
	lambda : np.random.uniform(40, 120), \
	lambda : np.random.uniform(-20, 80)],\
	[lambda x : pwl(x, [5, 35, 85], [.1, [.2, .5], [.5, .05], .9]), \
	lambda x : pwl(x, [50, 75, 100], [.4, [.3, .1], .15, .9]), \
	lambda x : pwl(x, [0, 20, 40], [.2, .3, [.4, .5], .6])])
