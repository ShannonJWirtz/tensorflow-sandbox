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
