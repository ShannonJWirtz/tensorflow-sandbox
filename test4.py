from gensource import *

obs = 600
category_values = ['mobile', 'desktop']
x_range = [0,100]

def mobile_probfunc(x):
	if x < 10:
		return 0.2
	if x < 30:
		return 0.05
	if x < 60:
		return 0.25
	
	return 0.15

def tablet_probfunc(x):
	if x < 20:
		return 0.8
	if x < 30:
		return 0.1
	if x < 60:
		return 0.70
	return 0.2

def desktop_probfunc(x):
	if x < 10:
		return 0.1
	if x < 20:
		return 0.05
	if x < 60:
		return 0.05 + (x - 20)*(0.7-0.05)/40
	return 0.70

def probfunc(x, cat):
	return {
		'mobile' : mobile_probfunc(x),
		'tablet' : tablet_probfunc(x),
		'desktop' : desktop_probfunc(x)
	}[cat]



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
	cat = np.random.choice(category_values)
	x = np.random.uniform(x_range[0], x_range[1])
	prob = probfunc(x, cat)
	outcome = np.random.binomial(1, prob)
	data.append({
		'category' : cat, 
		'x' : x, 
		'prob' : prob, 
		'outcome' : outcome})
data = pd.DataFrame(data)
data_preds = data.copy()

input_vars = ['x', 'category']
output_vars = ['prob']

input_data = data[input_vars]
output_data = data[output_vars]

input_scaling_vars = ['x']
input_encoding_vars = ['category']
input_encoding_drop_values = [data['category'].iloc[0]]
input_encoded_drop_columns = [a+'_'+b for a,b in zip(input_encoding_vars, input_encoding_drop_values)]
sscaler = preprocessing.StandardScaler()
scaling_vars_fit = sscaler.fit(input_data[input_scaling_vars])
input_data_scaled = input_data
scaling_vars_transformed = scaling_vars_fit.transform(input_data_scaled[input_scaling_vars])
input_data_scaled[input_scaling_vars] = scaling_vars_transformed

input_data_scaled_encoded_preselect = pd.get_dummies(input_data_scaled, columns = input_encoding_vars)

input_data_scaled_encoded = input_data_scaled_encoded_preselect.drop(input_encoded_drop_columns, axis = 1)


inputs_array = np.array(input_data_scaled_encoded)
outputs_array = np.array(output_data)

xt = tf.placeholder('float', [None,2])
yt = tf.placeholder('float', [None,1])

vars = paramr([2, 5, 5, 1], 'model')
net = NNet(vars, activations = [tf.nn.softplus,tf.nn.softplus, tf.nn.sigmoid])
pred = net.model(xt)

cost = tf.reduce_sum(tf.pow(tf.subtract(pred,yt),2))/len(inputs_array) #+ \
#tf.reduce_sum([tf.reduce_sum(tf.abs(var)) for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model')])

#cost = -tf.reduce_mean(yt*tf.log(pred) +(1-yt)*tf.log(1-pred))



with tf.Session() as sess:
	optimizeMethod = tf.train.GradientDescentOptimizer(0.1)
	optimizer = optimizeMethod.minimize(cost)
	init = tf.global_variables_initializer()
	sess.run(init)
	
	graphinputs = {xt: inputs_array, yt: outputs_array}

	training_cost = sess.run(cost, feed_dict = graphinputs)
	taste = sess.run(pred, feed_dict = graphinputs)
	precost1 = sess.run(tf.pow(tf.subtract(pred,yt),1), feed_dict = graphinputs)
	precost2 = sess.run(tf.pow(tf.subtract(pred,yt),2), feed_dict = graphinputs)
	print(training_cost, sep= '\n')

	for i in range(50000):
		sess.run(optimizer, feed_dict = graphinputs)
	
	print('adjusting... \n')

	training_cost = sess.run(cost, feed_dict = graphinputs)
	final_preds = sess.run(pred, feed_dict = {xt : inputs_array})


	print('Training Finished!')
	print(training_cost, sep= '\n')

	data_preds['preds'] = np.transpose(taste)[0]

grouped_data_pred = data_preds.groupby('category')

nrows = int(math.ceil(len(grouped_data_pred)/2.))

fig, axs = plt.subplots(nrows, 2)

for (name, df), ax in zip (grouped_data_pred, axs.flat):
    df.plot(x = 'x', y = 'prob', kind = 'scatter', ax = ax, color = 'b')
    df.plot(x = 'x', y = 'preds', kind = 'scatter', ax = ax, color = 'r')
    #df.plot(x = 'x', y = 'preds', kind = 'scatter', style = 'r--', ax = ax)
plt.show()

'''
	plotting_inputs = np.transpose()
	plt.plot(inputs_array, y, 'bs', x, np.transpose(taste)[0], 'b--')
	plt.legend()
	plt.show()

'''