#! /usr/bin/python

"""Create an LSTM network in torch and in tensorflow, and see what the weight sizes are
   If they're the same, then, let's see how we can convert one to the other.
   We'll start with a unidirectional LSTM, and then move to a bidirectional one.
"""

import sys
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import tensorflow as tf

def tfbatchnorm(inp_, weight, bias, running_mean, running_var):
	return ((inp_ - running_mean) / tf.sqrt(running_var+1e-12)) * weight + bias

def main():
	np.random.seed(12347)
	batchsz = 3
	inpsz = 5

	torchbn = nn.BatchNorm1d(inpsz)
	torchbn.eval()
# weight (32L,)
# bias (32L,)
# running_mean (32L,)
# running_var (32L,)

	tfpl_ = tf.placeholder(name="input", shape=(None, inpsz), dtype=tf.float32)
	tfweight = tf.get_variable(initializer=tf.zeros_initializer(), shape=(inpsz,), dtype=tf.float32, name="weight")
	tfbias = tf.get_variable(initializer=tf.zeros_initializer(), shape=(inpsz,), dtype=tf.float32, name="bias")
	tfrunning_mean = tf.get_variable(initializer=tf.zeros_initializer(), shape=(inpsz,), dtype=tf.float32, name="running_mean")
	tfrunning_var = tf.get_variable(initializer=tf.zeros_initializer(), shape=(inpsz,), dtype=tf.float32, name="running_var")
	tfbn = tfbatchnorm(tfpl_, tfweight, tfbias, tfrunning_mean, tfrunning_var)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		# weight = np.random.randn(inpsz)
		# bias = np.random.randn(inpsz)
		# running_mean = np.random.randn(inpsz)
		# running_var = np.power(np.random.randn(inpsz), 2)
		weight = np.ones((inpsz))
		bias = np.zeros((inpsz))
		running_mean = np.zeros((inpsz))
		running_var = np.ones((inpsz))

		torchbn.weight.data.copy_(torch.FloatTensor(weight))
		torchbn.bias.data.copy_(torch.FloatTensor(bias))
		torchbn.running_mean.copy_(torch.FloatTensor(running_mean))
		torchbn.running_var.copy_(torch.FloatTensor(running_var))

		sess.run(tf.assign(tfweight, weight))
		sess.run(tf.assign(tfbias, bias))
		sess.run(tf.assign(tfrunning_mean, running_mean))
		sess.run(tf.assign(tfrunning_var, running_var))

		inp_ = np.random.randn(batchsz, inpsz).astype(np.float32)
		torchout = torchbn(Variable(torch.FloatTensor(inp_)))
		tfout = sess.run(tfbn, feed_dict={tfpl_: inp_})
		print np.sum(np.power(tfout - torchout.data.cpu().numpy(), 2))

if __name__ == "__main__":
	main()
