from tfsource import *
from stocgenfuncs1 import *
from aggtransfuncs import *

s2 = \
simple_model_1([0.45,0.1,0.45], \
	[lambda : np.random.uniform(0, 100), \
	lambda : np.random.uniform(40, 120), \
	lambda : np.random.uniform(-20, 80)],\
	[lambda x : pwl(x, [5, 35, 85], [.1, [.2, .5], [.5, .05], .9]), \
	lambda x : pwl(x, [50, 75, 100], [.4, [.3, .1], .15, .9]), \
	lambda x : pwl(x, [0, 20, 40], [.2, .3, [.4, .5], .6])])

a = Agg(s2, 500, 0.8, 'x', 'category', 'outcome')

xt = tf.placeholder('float', [None,4])
yt = tf.placeholder('float', [None,1])

vars = paramr([4,2,1], 'model')
net = NNet(vars, activations = [tf.nn.softplus, tf.sigmoid])
pred = net.model(xt)

reg_coeff = 0.0001
cost = -tf.reduce_mean(yt*tf.log(pred) +(1-yt)*tf.log(1-pred))+ \
reg_coeff*tf.reduce_sum([tf.reduce_sum(tf.abs(var)) for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model')])

t = Tftrain(a.train_inputs_array, xt, yt, a.train_outputs_array,pred, cost, 100, 500)




'''
class DataGen(object):

	def __init__(self, ):








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
'''