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

def tflstmfunc(layers, hiddensz, inpsz):
# Perhaps this needs to be manipulated to behave like torch's bidirectional RNN
# Is it that tensorflow is only bidirectional on the first layer, but
# torch is bidirectional on _every_ layer?
	with tf.variable_scope("inp"):
		inp_ = tf.placeholder(name="input", shape=(None, None, inpsz), dtype=tf.float32)

	with tf.variable_scope("lstm"):
		output = inp_
		for lnum in xrange(0, layers):
			with tf.variable_scope("layer_%d" % (lnum,)):
				forward_prototype = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(hiddensz, forget_bias=0.0)])
				backward_prototype = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(hiddensz, forget_bias=0.0)])
				output = tf.concat(tf.nn.bidirectional_dynamic_rnn(forward_prototype, backward_prototype, output, dtype=tf.float32)[0], axis=2)
	return inp_, output

def initializer(*dims):
	return np.random.randn(*dims).astype(np.float32)

def rearrange(torcharray):
	hiddensz = torcharray.shape[0] / 4

	# This order was _painstakingly_ found, by trying all possible combinations.
	w_i = torcharray[0*hiddensz:1*hiddensz]
	w_f = torcharray[2*hiddensz:3*hiddensz]
	w_g = torcharray[1*hiddensz:2*hiddensz]
	w_o = torcharray[3*hiddensz:4*hiddensz]

	return np.concatenate([w_i, w_f, w_g, w_o], axis=0)

#pylint: disable=too-many-locals
def main():
	np.random.seed(12347)
	batchsz = 5
	layers = 3
	inpsz = 7
	hiddensz = 11
	sequencesz = 12


	torchlstm = nn.LSTM(input_size=inpsz, hidden_size=hiddensz, num_layers=layers, batch_first=True, bidirectional=True)
# weight_ih_l0 (44L, 7L)
# weight_hh_l0 (44L, 11L)
# bias_ih_l0 (44L,)
# bias_hh_l0 (44L,)
# weight_ih_l0_reverse (44L, 7L)
# weight_hh_l0_reverse (44L, 11L)
# bias_ih_l0_reverse (44L,)
# bias_hh_l0_reverse (44L,)
# weight_ih_l1 (44L, 22L)
# weight_hh_l1 (44L, 11L)
# bias_ih_l1 (44L,)
# bias_hh_l1 (44L,)
# weight_ih_l1_reverse (44L, 22L)
# weight_hh_l1_reverse (44L, 11L)
# bias_ih_l1_reverse (44L,)
# bias_hh_l1_reverse (44L,)
# weight_ih_l2 (44L, 22L)
# weight_hh_l2 (44L, 11L)
# bias_ih_l2 (44L,)
# bias_hh_l2 (44L,)
# weight_ih_l2_reverse (44L, 22L)
# weight_hh_l2_reverse (44L, 11L)
# bias_ih_l2_reverse (44L,)
# bias_hh_l2_reverse (44L,)

	tfinp_, tflstm_ = tflstmfunc(layers, hiddensz, inpsz)
# lstm/layer_0/bidirectional_rnn/fw/multi_rnn_cell/cell_0/lstm_cell/kernel:0 (18, 44)
# lstm/layer_0/bidirectional_rnn/fw/multi_rnn_cell/cell_0/lstm_cell/bias:0 (44,)
# lstm/layer_0/bidirectional_rnn/bw/multi_rnn_cell/cell_0/lstm_cell/kernel:0 (18, 44)
# lstm/layer_0/bidirectional_rnn/bw/multi_rnn_cell/cell_0/lstm_cell/bias:0 (44,)
# lstm/layer_1/bidirectional_rnn/fw/multi_rnn_cell/cell_0/lstm_cell/kernel:0 (33, 44)
# lstm/layer_1/bidirectional_rnn/fw/multi_rnn_cell/cell_0/lstm_cell/bias:0 (44,)
# lstm/layer_1/bidirectional_rnn/bw/multi_rnn_cell/cell_0/lstm_cell/kernel:0 (33, 44)
# lstm/layer_1/bidirectional_rnn/bw/multi_rnn_cell/cell_0/lstm_cell/bias:0 (44,)
# lstm/layer_2/bidirectional_rnn/fw/multi_rnn_cell/cell_0/lstm_cell/kernel:0 (33, 44)
# lstm/layer_2/bidirectional_rnn/fw/multi_rnn_cell/cell_0/lstm_cell/bias:0 (44,)
# lstm/layer_2/bidirectional_rnn/bw/multi_rnn_cell/cell_0/lstm_cell/kernel:0 (33, 44)
# lstm/layer_2/bidirectional_rnn/bw/multi_rnn_cell/cell_0/lstm_cell/bias:0 (44,)

	with tf.Session() as sess:
		# So, it looks like lstm/cell_{n}/lstm_cell/kernel:0 is concat([weight_ih_l{n}, weight_hh_l{n}], dim=1).transpose()
		# and lstm/cell_{n}/lstm_cell/bias:0 is concat([bias_ih_l{n}, bias_hh_l{n}], dim=0).transpose()
		torchparams = {}
		for lnum in xrange(0, layers):
			torchparams["weight_ih_l%d" % (lnum,)] = initializer(4 * hiddensz, inpsz if lnum == 0 else 2 * hiddensz).astype(np.float32)
			torchparams["weight_hh_l%d" % (lnum,)] = initializer(4 * hiddensz, hiddensz).astype(np.float32)
			torchparams["bias_ih_l%d" % (lnum,)] = initializer(4 * hiddensz)
			torchparams["bias_hh_l%d" % (lnum,)] = initializer(4 * hiddensz)
			torchparams["weight_ih_l%d_reverse" % (lnum,)] = initializer(4 * hiddensz, inpsz if lnum == 0 else 2 * hiddensz).astype(np.float32)
			torchparams["weight_hh_l%d_reverse" % (lnum,)] = initializer(4 * hiddensz, hiddensz).astype(np.float32)
			torchparams["bias_ih_l%d_reverse" % (lnum,)] = initializer(4 * hiddensz)
			torchparams["bias_hh_l%d_reverse" % (lnum,)] = initializer(4 * hiddensz)

		tfparams = {}
		order = ["ih", "hh"]
		for lnum in xrange(0, layers):
			tfparams["lstm/layer_%d/bidirectional_rnn/fw/multi_rnn_cell/cell_0/lstm_cell/kernel:0" % (lnum,)] = np.concatenate([rearrange(torchparams["weight_%s_l%d" % (order[0], lnum)]), rearrange(torchparams["weight_%s_l%d" % (order[1], lnum)])], axis=1).transpose(1, 0)
			tfparams["lstm/layer_%d/bidirectional_rnn/fw/multi_rnn_cell/cell_0/lstm_cell/bias:0" % (lnum,)] = rearrange(torchparams["bias_%s_l%d" % (order[0], lnum)]) + rearrange(torchparams["bias_%s_l%d" % (order[1], lnum)])
			tfparams["lstm/layer_%d/bidirectional_rnn/bw/multi_rnn_cell/cell_0/lstm_cell/kernel:0" % (lnum,)] = np.concatenate([rearrange(torchparams["weight_%s_l%d_reverse" % (order[0], lnum)]), rearrange(torchparams["weight_%s_l%d_reverse" % (order[1], lnum)])], axis=1).transpose(1, 0)
			tfparams["lstm/layer_%d/bidirectional_rnn/bw/multi_rnn_cell/cell_0/lstm_cell/bias:0" % (lnum,)] = rearrange(torchparams["bias_%s_l%d_reverse" % (order[0], lnum)]) + rearrange(torchparams["bias_%s_l%d_reverse" % (order[1], lnum)])

		for name, param in torchlstm.named_parameters():
			param.data.copy_(torch.FloatTensor(torchparams[name]))

		for var in tf.trainable_variables():
			sys.stderr.write("%s\n" % (var.name))
			sess.run(tf.assign(var, tfparams[var.name]))

		inp = np.random.randn(batchsz, sequencesz, inpsz).astype(np.float32)
		inptorch = Variable(torch.FloatTensor(inp))
		torchout = torchlstm(inptorch)[0].data.cpu().numpy()
		tfout = sess.run(tflstm_, feed_dict={tfinp_: inp})
		print torchout.shape
		print tfout.shape
		print np.power(torchout - tfout, 2).sum()
#pylint: enable=too-many-locals


if __name__ == "__main__":
	main()
