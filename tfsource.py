import tensorflow as tf
import numpy as np
from random import shuffle
import math

class NNet(object):
	def __init__(self, params, activations):
		self.params = params
		self.hidden_layers_number = len(params)
		self.activations = []
		if callable(activations):
			for i in range(self.hidden_layers_number):
				self.activations.append(activations)
		elif type(activations) == list:
			for i in range(self.hidden_layers_number):
				self.activations.append(activations[i])
		else:
			print("Error, activations must be a list or a single function")
		
	def layer(self, inputs, params, activation = tf.nn.sigmoid):
		linear_transform = tf.matmul(inputs, params['weights'])
		affine_transform = tf.add(linear_transform, params['bias'])
		output = activation(affine_transform)
		return output

	def model(self, inputs):
		output = inputs
		for i in range(self.hidden_layers_number):
			output = self.layer(output, self.params[i], self.activations[i])
		return output

def randn(shape, sd = 1):
	return tf.Variable(tf.random_normal(shape = shape, stddev = sd))

def paramr(layers, scopename):
	with tf.variable_scope(scopename):
		output = []
		for i in range(len(layers)-1):
			output.append({'weights' : randn([layers[i], layers[i+1]]), 'bias' : randn([layers[i+1]])})
	return output


#
#optimizer
#optimizer arguments
#traindata
#model
#loss function

#batch size
#epochs or finish condition
#

class Tftrain(object):

	def __init__(self, traininputs, trX, trY, trainoutputs, model, cost,\
		batchsize, epochs, optimizer = tf.train.AdamOptimizer,\
		optimizer_arguments = {}):

		self.traininputs = traininputs
		self.trainoutputs = trainoutputs
		self.optimizer_arguments = optimizer_arguments
		self.model = model
		self.cost = cost
		self.batchsize = batchsize
		self.epochs = epochs
		self.inputslen = len(traininputs)

		self.sess = tf.Session()
		optimizeMethod = tf.train.AdamOptimizer()
		self.optimizer = optimizeMethod.minimize(cost)
		init = tf.global_variables_initializer()
		self.sess.run(init)
		self.stc = self.sess.run(self.cost,{trX : self.traininputs, trY : self.trainoutputs})
		for i in range(epochs):
			indices = np.random.choice(self.inputslen, self.inputslen, replace = False)
			lobs = 0
			uobs = batchsize
			while lobs < self.inputslen:
				bindices = indices[lobs:uobs]
				self.sess.run(self.optimizer,{trX : self.traininputs[bindices], trY : self.trainoutputs[bindices]})
				lobs = lobs + self.batchsize
				uobs = uobs + self.batchsize
		self.etc = self.sess.run(self.cost,{trX : self.traininputs, trY : self.trainoutputs})

	def pred(self, inputsarray, tx):
		output = self.sess.run(self.model, {tx : inputsarray})
		return output

'''
		with tf.Session as sess:
			self.optimizeMethod = tf.train.AdamOptimizer
			self.optimizer = self.optimizeMethod.minimize(cost)
			init = tf.global_variables_initializer()
			sess.run(init)

			self.starttraincost = sess.run(self.optimizer,{trX : self.traininputs, trY : self.trainoutputs})

			for i in range(epochs):
				indices = np.random.choice(self.inputslen, self.inputslen, replace = False)
				lobs = 0
				uobs = batsize - 1
				while lobs < self.inputslen:
					bindicies = indicies[lobs:uobs]
					xt = self.traininputs[bindicies]
					yt = self.trainoutputs[bindicies]
					sess.run(self.optimizer,{trX : xt, trX : yt})
					lobs = lobs + self.batchsize
					uobs = uobs + self.batchsize

			self.endtraincost = sess.run(self.optimizer,{trX : self.traininputs, trY : self.trainoutputs})

'''






'''
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
'''