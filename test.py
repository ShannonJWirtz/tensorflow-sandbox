from gensource import *

x = np.linspace(-10, 10,10)
y = x*2 + 3

xt = tf.placeholder('float', [None,1])
yt = tf.placeholder('float', [None,1])

vars = paramr([1,2,1], 'model')
net = NNet(vars, activations = [tf.nn.softplus, tf.identity])
pred = net.model(xt)

cost = tf.reduce_sum(tf.pow(tf.subtract(pred,yt),2))/len(x) + \
tf.reduce_sum([tf.reduce_sum(tf.abs(var)) for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model')])


with tf.Session() as sess:
	optimizeMethod = tf.train.AdamOptimizer(0.1)
	optimizer = optimizeMethod.minimize(cost)
	init = tf.global_variables_initializer()
	sess.run(init)
	xtrans = np.transpose([x])
	ytrans = np.transpose([y])
	graphinputs = {xt: xtrans, yt: ytrans}

	training_cost = sess.run(cost, feed_dict = {xt: xtrans, yt: ytrans})
	taste = sess.run(pred, feed_dict = {xt:xtrans})
	precost1 = sess.run(tf.pow(tf.subtract(pred,yt),1), feed_dict = graphinputs)
	precost2 = sess.run(tf.pow(tf.subtract(pred,yt),2), feed_dict = graphinputs)
	print(xtrans, ytrans, taste, precost1, precost2, training_cost, sep= '\n')

	for i in range(100):
		sess.run(optimizer, feed_dict = graphinputs)
	
	print('adjusting... \n')

	training_cost = sess.run(cost, feed_dict = {xt: xtrans, yt: ytrans})
	taste = sess.run(pred, feed_dict = {xt:xtrans})
	precost1 = sess.run(tf.pow(tf.subtract(pred,yt),1), feed_dict = graphinputs)
	precost2 = sess.run(tf.pow(tf.subtract(pred,yt),2), feed_dict = graphinputs)
	print(xtrans, ytrans, taste, precost1, precost2, training_cost, sep= '\n')

	print('Training Finished!')
	print(np.transpose(taste))
	plt.plot(x, y, 'bs', x, np.transpose(taste)[0], 'b--')
	plt.legend()
	plt.show()