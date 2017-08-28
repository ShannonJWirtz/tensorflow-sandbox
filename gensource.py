from tfsource import *
from datasource import *
from detgenfuncs1 import *
from stocgenfuncs1 import *

obs = 100
category_values = ['mobile', 'desktop', 'tablet']
x_min = 0; x_max = 200; x_range_length = x_max - x_min


def probfunc(x, cat):
	if cat == 'mobile':
		return pwl(x, \
			[x_min + bp*x_range_length/100 for bp in [5, 15, 40, 75, 90]], \
			[0.05, 0.2, 0.3, 0.05, 0.7, 0.2])
	if cat == 'desktop':
		return pwl(x, \
			[x_min + bp*x_range_length/100 for bp in [25, 50, 75]], \
			[0.9, 0.3, 0.4, 0.85])
	if cat == 'tablet':
		return pwl(x, \
			[x_min + bp*x_range_length/100 for bp in [60]], \
			[0.5, 0.6])

data = []

for i in range(obs):
	cat = np.random.choice(category_values)
	x = np.random.uniform(x_min, x_max)
	prob = probfunc(x, cat)
	outcome = np.random.binomial(1, prob)
	data.append({
		'category' : cat, 
		'x' : x, 
		'prob' : prob, 
		'outcome' : outcome})
data = pd.DataFrame(data)