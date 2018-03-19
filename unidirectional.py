#! /usr/bin/python

"""Create an LSTM network in torch and in tensorflow, and see what the weight sizes are
   If they're the same, then, let's see how we can convert one to the other.
   We'll start with a unidirectional LSTM, and then move to a bidirectional one.
"""

import random
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import tensorflow as tf

def tflstmfunc(layers, hiddensz, inpsz):
	with tf.variable_scope("inp"):
		inp_ = tf.placeholder(name="input", shape=(None, None, inpsz), dtype=tf.float32)

	with tf.variable_scope("lstm"):
		stacks = [tf.nn.rnn_cell.LSTMCell(hiddensz, forget_bias=0.0) for _ in xrange(0, layers)]
		prototype = tf.nn.rnn_cell.MultiRNNCell(stacks)
		initial_state = prototype.zero_state(tf.shape(inp_)[0], dtype=tf.float32)
		output = tf.nn.dynamic_rnn(prototype, inp_, initial_state=initial_state, dtype=tf.float32)
	return inp_, output

def initializer(*dims):
	return np.random.randn(*dims).astype(np.float32)

def rearrange(torcharray):
	hiddensz = torcharray.shape[0] / 4

	order = [0, 2, 1, 3] # 2.6038472e-14 !!

	A_i = torcharray[0*hiddensz:1*hiddensz]
	A_f = torcharray[2*hiddensz:3*hiddensz]
	A_g = torcharray[1*hiddensz:2*hiddensz]
	A_o = torcharray[3*hiddensz:4*hiddensz]

	return np.concatenate([A_i, A_f, A_g, A_o], axis=0)

def main():
	np.random.seed(12347)
	batchsz = 5
	layers = 3
	inpsz = 7
	hiddensz = 11
	sequencesz = 12


	torchlstm = nn.LSTM(input_size=inpsz, hidden_size=hiddensz, num_layers=layers, batch_first=True)

# weight_ih_l0 (44L, 7L)
# weight_hh_l0 (44L, 11L)
# bias_ih_l0 (44L,)
# bias_hh_l0 (44L,)
# weight_ih_l1 (44L, 11L)
# weight_hh_l1 (44L, 11L)
# bias_ih_l1 (44L,)
# bias_hh_l1 (44L,)
# weight_ih_l2 (44L, 11L)
# weight_hh_l2 (44L, 11L)
# bias_ih_l2 (44L,)
# bias_hh_l2 (44L,)
	tfinp_, tflstm_ = tflstmfunc(layers, hiddensz, inpsz)
# lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel:0 (18, 44)
# lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/bias:0 (44,)
# lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel:0 (22, 44)
# lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/bias:0 (44,)
# lstm/rnn/multi_rnn_cell/cell_2/lstm_cell/kernel:0 (22, 44)
# lstm/rnn/multi_rnn_cell/cell_2/lstm_cell/bias:0 (44,)


	with tf.Session() as sess:
		# So, it looks like lstm/cell_{n}/lstm_cell/kernel:0 is concat([weight_ih_l{n}, weight_hh_l{n}], dim=1).transpose()
		# and lstm/cell_{n}/lstm_cell/bias:0 is concat([bias_ih_l{n}, bias_hh_l{n}], dim=0).transpose()
		torchparams = {}
		for lnum in xrange(0, layers):
			torchparams["weight_ih_l%d" % (lnum,)] = initializer(4 * hiddensz, inpsz if lnum == 0 else hiddensz).astype(np.float32)
			torchparams["weight_hh_l%d" % (lnum,)] = initializer(4 * hiddensz, hiddensz).astype(np.float32)
			torchparams["bias_ih_l%d" % (lnum,)] = initializer(4 * hiddensz)
			torchparams["bias_hh_l%d" % (lnum,)] = initializer(4 * hiddensz)

		tfparams = {}
		order = ["ih", "hh"]
		for lnum in xrange(0, layers):
			tfparams["lstm/rnn/multi_rnn_cell/cell_%d/lstm_cell/kernel:0" % (lnum,)] = np.concatenate([rearrange(torchparams["weight_%s_l%d" % (order[0], lnum)]), rearrange(torchparams["weight_%s_l%d" % (order[1], lnum)])], axis=1).transpose(1, 0)
			tfparams["lstm/rnn/multi_rnn_cell/cell_%d/lstm_cell/bias:0" % (lnum,)] = rearrange(torchparams["bias_%s_l%d" % (order[0], lnum)]) + rearrange(torchparams["bias_%s_l%d" % (order[1], lnum)])

		for name, param in torchlstm.named_parameters():
			param.data.copy_(torch.FloatTensor(torchparams[name]))

		for var in tf.trainable_variables():
			sess.run(tf.assign(var, tfparams[var.name]))

		inp = np.random.randn(batchsz, sequencesz, inpsz).astype(np.float32)
		inptorch = Variable(torch.FloatTensor(inp))
		torchout = torchlstm(inptorch)[0].data.cpu().numpy()
		tfout = sess.run(tflstm_, feed_dict={tfinp_: inp})[0]
		print np.power(torchout - tfout, 2).sum()


if __name__ == "__main__":
	main()


