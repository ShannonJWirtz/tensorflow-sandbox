from gensource import *

obs = 2000
x_range = [0,100]
'''
def probfunc(x):
	if x < 10:
		return 0.1
	if x < 20:
		return 0.05
	if x < 60:
		return 0.05 + (x - 20)*(0.7-0.05)/40
	return 0.70
'''
def probfunc(x):
	upper = x_range[1]
	if x < upper/2:
		return(x/upper)
	return 1-x/upper


'''
data = np.array([])

for i in range(obs):
	cat = np.random.choice(category_values)
	x = np.random.uniform(x_range[0], x_range[1])
	prob = probfunc(x, cat)
	outcome = np.random.binomial(1, prob)
	data.append([cat, x, prob, outcome])

y = [k[range(len(k)),-1]
'''
data = []

for i in range(obs):
	x = np.random.uniform(x_range[0], x_range[1])
	prob = probfunc(x)
	outcome = np.random.binomial(1, prob)
	data.append({
		'x' : x, 
		'prob' : prob, 
		'outcome' : outcome})
data = pd.DataFrame(data)
data_preds = data.copy()

input_vars = ['x']
output_vars = ['outcome']

input_data = data[input_vars]
output_data = data[output_vars]

input_scaling_vars = ['x']
sscaler = preprocessing.StandardScaler()
scaling_vars_fit = sscaler.fit(input_data[input_scaling_vars])
input_data_scaled = input_data
scaling_vars_transformed = scaling_vars_fit.transform(input_data_scaled[input_scaling_vars])
input_data_scaled[input_scaling_vars] = scaling_vars_transformed

inputs_array = np.array(input_data_scaled)
outputs_array = np.array(output_data)

xt = tf.placeholder('float', [None,1])
yt = tf.placeholder('float', [None,1])

vars = paramr([1, 6, 6, 1], 'model')
net = NNet(vars, activations = [tf.nn.softplus,tf.nn.softplus, tf.nn.sigmoid])
pred = net.model(xt)

#cost = tf.reduce_sum(tf.pow(tf.subtract(pred,yt),2))/len(inputs_array) #+ \
#tf.reduce_sum([tf.reduce_sum(tf.abs(var)) for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model')])

reg_coeff = 0.0001
cost = -tf.reduce_mean(yt*tf.log(pred) +(1-yt)*tf.log(1-pred))+ \
reg_coeff*tf.reduce_sum([tf.reduce_sum(tf.abs(var)) for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model')])




with tf.Session() as sess:
	optimizeMethod = tf.train.AdamOptimizer()
	optimizer = optimizeMethod.minimize(cost)
	init = tf.global_variables_initializer()
	sess.run(init)
	
	graphinputs = {xt: inputs_array, yt: outputs_array}

	training_cost = sess.run(cost, feed_dict = graphinputs)
	taste = sess.run(pred, feed_dict = graphinputs)
	precost1 = sess.run(tf.pow(tf.subtract(pred,yt),1), feed_dict = graphinputs)
	precost2 = sess.run(tf.pow(tf.subtract(pred,yt),2), feed_dict = graphinputs)
	print(training_cost, sep= '\n')

	for i in range(100000):
		sess.run(optimizer, feed_dict = graphinputs)
	
	print('adjusting... \n')
	training_cost = sess.run(cost, feed_dict = graphinputs)
	final_preds = sess.run(pred, feed_dict = {xt : inputs_array})

	precost = sess.run(-(yt*tf.log(pred) + (1-yt)*tf.log(1-pred)), feed_dict = graphinputs)

	print('Training Finished!')
	print(training_cost, sep= '\n')
	data_preds['precost'] = np.transpose(precost)[0]
	data_preds['preds'] = np.transpose(final_preds)[0]

data_preds.plot('x', 'prob', 'scatter', color = 'b')
data_preds.plot('x', 'preds', 'scatter', color = 'r')
plt.show()


'''
	plotting_inputs = np.transpose()
	plt.plot(inputs_array, y, 'bs', x, np.transpose(taste)[0], 'b--')
	plt.legend()
	plt.show()

'''