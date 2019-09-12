# -*- coding: utf-8 -*-

__author__ = 'Wenzhi Mao'

__all__ = ['buildNet']


def makeOp(inputs, whichop, chanel_num, name, l2, weights=None, layerdilate=1, rncn=False):
	import tensorflow as tf
	if weights is None:
		weights={}
	whichtype = whichop.split('_')[0]
	whichpara = whichop.split('_')[1:]
	if whichtype == 'identity':
		if rncn:
			return tf.identity(inputs[:,:,:,:tf.shape(inputs)[3]//3], name=name)
		else:
			return tf.identity(inputs, name=name)
	elif whichtype == 'conv':
		return tf.layers.conv2d(inputs, filters=chanel_num,
			kernel_size=(int(whichpara[0]), int(whichpara[0])),
			padding='same',
			dilation_rate=(layerdilate, layerdilate),
			activation=tf.nn.leaky_relu,
			kernel_initializer=getWeight(weights,"kernel:0"),
			bias_initializer=getWeight(weights,"bias:0"),
			kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2),
			name=name)
	elif whichtype == 'sep':
		return tf.layers.separable_conv2d(inputs, filters=chanel_num,
			kernel_size=(int(whichpara[0]), int(whichpara[0])),
			padding='same',
			dilation_rate=(layerdilate, layerdilate),
			activation=tf.nn.leaky_relu,
			depthwise_initializer=getWeight(weights,"depthwise_kernel:0"),
			pointwise_initializer=getWeight(weights,"pointwise_kernel:0"),
			bias_initializer=getWeight(weights,"bias:0"),
			depthwise_regularizer=tf.contrib.layers.l2_regularizer(scale=l2),
			pointwise_regularizer=tf.contrib.layers.l2_regularizer(scale=l2),
			bias_regularizer=tf.contrib.layers.l2_regularizer(scale=l2),
			bias_constraint=None,
			name=name)
	elif whichtype == 'ave':
		return tf.layers.average_pooling2d(inputs,
			pool_size=int(whichpara[0]),
			strides=(layerdilate, layerdilate),
			padding='same',
			name=name)
	elif whichtype == 'min':
		with tf.variable_scope(name):
			return tf.negative(tf.layers.max_pooling2d(-inputs,
				pool_size=int(whichpara[0]),
				strides=(layerdilate,layerdilate),
				padding='same',))
	elif whichtype == 'max':
		return tf.layers.max_pooling2d(inputs,
			pool_size=int(whichpara[0]),
			strides=(layerdilate,layerdilate),
			padding='same',
			name=name)
	elif whichtype == 'dil-sep':
		return tf.layers.separable_conv2d(inputs, filters=chanel_num,
			kernel_size=(3, 3),
			padding='same',
			dilation_rate=(int(whichpara[0])*layerdilate, int(whichpara[0])*layerdilate),
			activation=tf.nn.leaky_relu,
			depthwise_initializer=getWeight(weights,"depthwise_kernel:0"),
			pointwise_initializer=getWeight(weights,"pointwise_kernel:0"),
			bias_initializer=getWeight(weights,"bias:0"),
			depthwise_regularizer=tf.contrib.layers.l2_regularizer(scale=l2),
			pointwise_regularizer=tf.contrib.layers.l2_regularizer(scale=l2),
			bias_regularizer=tf.contrib.layers.l2_regularizer(scale=l2),
			bias_constraint=None,
			name=name)
	elif whichtype == '2step':
		with tf.variable_scope(name):
			temp1 = tf.layers.conv2d(inputs, filters=chanel_num,
				kernel_size=(int(whichpara[0].split('x')[0]), int(whichpara[0].split('x')[1])),
				padding='same',
				dilation_rate=(layerdilate, layerdilate),
				activation=tf.nn.leaky_relu,
				kernel_initializer=getWeight(weights,"conv2d/kernel:0"),
				bias_initializer=getWeight(weights,"conv2d/bias:0"),
				kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2),)
			return tf.layers.conv2d(temp1, filters=chanel_num,
				kernel_size=(int(whichpara[1].split('x')[0]), int(whichpara[1].split('x')[1])),
				padding='same',
				dilation_rate=(layerdilate, layerdilate),
				activation=tf.nn.leaky_relu,
				kernel_initializer=getWeight(weights,"conv2d_1/kernel:0"),
				bias_initializer=getWeight(weights,"conv2d_1/bias:0"),
				kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2),)
	elif whichtype == 'dia':
		return tf.layers.conv2d(inputs, filters=chanel_num,
			kernel_size=(int(whichpara[0]), int(whichpara[0])),
			padding='same',
			dilation_rate=(int(whichpara[1])*layerdilate, int(whichpara[1])*layerdilate),
			activation=tf.nn.leaky_relu,
			kernel_initializer=getWeight(weights,"kernel:0"),
			bias_initializer=getWeight(weights,"bias:0"),
			kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2),
			name=name)
	else:
		raise ValueError("Cannot find Op type.")
def RNCN(x):
	import tensorflow as tf
	with tf.variable_scope('RNCN'):
		mean, var = tf.nn.moments(x, [1], keep_dims=True)
		new_tensor1 = tf.div(tf.subtract(x, mean), tf.sqrt(var))
		mean, var = tf.nn.moments(x, [2], keep_dims=True)
		new_tensor2 = tf.div(tf.subtract(x, mean), tf.sqrt(var))
		return tf.concat([x,new_tensor1,new_tensor2],axis=3)
def buildLayer(prev, prev_prev, structure, name, l2, weights=None, layerdilate=1, rncn=False):
	import tensorflow as tf
	if weights is None:
		weights={}
	mapdict = {'h_i': prev, 'h_i-1': prev_prev}
	chanel_num = structure.channel_depth
	with tf.variable_scope("Hidden_{}".format(name)):
		weights1={'/'.join(i.split('/')[1:]):weights[i] for i in weights if i.split('/')[0]=="Hidden_{}".format(name)}
		for i in range(structure.combination_number):
			info = structure.structure[i]
			weights2={'/'.join(ii.split('/')[1:]):weights1[ii] for ii in weights1 if ii.split('/')[0]=="Inter_{}_1".format(i)}
			val1 = makeOp(mapdict[info[0][0]], info[0][1], chanel_num, name="Inter_{}_1".format(i), l2=l2, weights=weights2, layerdilate=layerdilate, rncn=rncn)
			weights2={'/'.join(ii.split('/')[1:]):weights1[ii] for ii in weights1 if ii.split('/')[0]=="Inter_{}_2".format(i)}
			val2 = makeOp(mapdict[info[1][0]], info[1][1], chanel_num, name="Inter_{}_2".format(i), l2=l2, weights=weights2, layerdilate=layerdilate, rncn=rncn)
			temp = tf.add(val1, val2, name="Inter_{}".format(i))
			if rncn:
				temp=RNCN(temp)
			mapdict['inter_{}'.format(i)] = temp
		pre_final = tf.concat([mapdict['{}'.format(i)] for i in structure.unused], axis=3, name='Pre-Final')
		final = tf.layers.conv2d(pre_final, filters=structure.channel_depth,
								 kernel_size=(1, 1),
								 padding='same',
								 dilation_rate=(1, 1),
								 activation=tf.nn.leaky_relu,
								 kernel_initializer=getWeight(weights1,'Final/kernel:0'),
								 bias_initializer=getWeight(weights1,'Final/bias:0'),
								 kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2),
								 name='Final')
		if rncn:
			final = RNCN(final)
	return final

def getWeight(weights,name):
	import tensorflow as tf
	if name not in weights:
		return tf.truncated_normal_initializer(stddev=.1)
	return tf.constant_initializer(weights[name])

def buildNet(input_chanel_num, structure, lr=1e-3, l2=0.,initweight=None, layerdilate=1, rncn=False, splitdevice=1):
	import tensorflow as tf
	from .layer import WebLayerStructure
	if not isinstance(structure, WebLayerStructure):
		raise ValueError("The `structure` is not a valid structure.")

	x = tf.placeholder("float", shape=[None, None, None, input_chanel_num], name='Input')
	y = tf.placeholder("float", shape=[None, None, None, 1], name='Label')
	if initweight is None:
		initweight={}
	istrain = tf.placeholder("bool")

	layers = []
	if splitdevice!=1:
		with tf.device('/device:GPU:1'):
			step0=tf.layers.conv2d(x, filters=structure.channel_depth,
							kernel_size=(1,1),
							padding='same',
							dilation_rate=(1,1),
							activation=tf.nn.leaky_relu,
							kernel_initializer=getWeight(initweight,'Hidden0/kernel:0'),
							bias_initializer=getWeight(initweight,'Hidden0/bias:0'),
							kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2),
							name='Hidden0')
			if rncn:
				step0=RNCN(step0)
	else:
		step0=tf.layers.conv2d(x, filters=structure.channel_depth,
							kernel_size=(1,1),
							padding='same',
							dilation_rate=(1,1),
							activation=tf.nn.leaky_relu,
							kernel_initializer=getWeight(initweight,'Hidden0/kernel:0'),
							bias_initializer=getWeight(initweight,'Hidden0/bias:0'),
							kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2),
							name='Hidden0')
		if rncn:
			step0=RNCN(step0)
	import math 
	for i in range(structure.stacking_depth):
		if splitdevice==1:
			if i == 0:
				layers.append(buildLayer(prev=step0, prev_prev=step0, structure=structure, name=i+1, l2=l2, weights=initweight, layerdilate=1, rncn=rncn))
			elif i == 1:
				layers.append(buildLayer(prev=layers[0], prev_prev=step0, structure=structure, name=i+1, l2=l2, weights=initweight, layerdilate=layerdilate, rncn=rncn))
			else:
				layers.append(buildLayer(prev=layers[i-1], prev_prev=layers[i-2], structure=structure, name=i+1, l2=l2, weights=initweight, layerdilate=layerdilate**i, rncn=rncn))
		else:
			with tf.device('/device:GPU:1'):
				if i == 0:
					layers.append(buildLayer(prev=step0, prev_prev=step0, structure=structure, name=i+1, l2=l2, weights=initweight, layerdilate=1, rncn=rncn))
				elif i == 1:
					layers.append(buildLayer(prev=layers[0], prev_prev=step0, structure=structure, name=i+1, l2=l2, weights=initweight, layerdilate=layerdilate, rncn=rncn))
				else:
					layers.append(buildLayer(prev=layers[i-1], prev_prev=layers[i-2], structure=structure, name=i+1, l2=l2, weights=initweight, layerdilate=layerdilate**i, rncn=rncn))
	if splitdevice==1:
		dropout = tf.layers.dropout(layers[structure.stacking_depth-1],
									rate=0.5,
									training=istrain,
									name='Dropout')
		pred = tf.layers.conv2d(dropout, filters=1,
			kernel_size=(1, 1),
			padding='same',
			dilation_rate=(1, 1),
			activation=None,
			kernel_initializer=getWeight(initweight,'Logit/kernel:0'),
			bias_initializer=getWeight(initweight,'Logit/bias:0'),
			kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2),
			name='Logit')
	else:
		with tf.device('/device:GPU:1'):
			dropout = tf.layers.dropout(layers[structure.stacking_depth-1],
										rate=0.5,
										training=istrain,
										name='Dropout')
			pred = tf.layers.conv2d(dropout, filters=1,
				kernel_size=(1, 1),
				padding='same',
				dilation_rate=(1, 1),
				activation=None,
				kernel_initializer=getWeight(initweight,'Logit/kernel:0'),
				bias_initializer=getWeight(initweight,'Logit/bias:0'),
				kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2),
				name='Logit')
	realpred = tf.sigmoid(pred, name='Prediction')
	loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=y, logits=pred)

	with tf.device('/device:GPU:0'):
		train_step = tf.contrib.layers.optimize_loss(
			loss=loss,
			global_step=tf.train.get_global_step(),
			learning_rate=lr,
			optimizer="Adam",
			name='train_step',)

	# with tf.device('/device:GPU:0'):
	# 	opt = tf.train.AdamOptimizer(learning_rate=lr)


	# 	gradient_all = opt.compute_gradients(loss) 


	# # grads_and_vars = opt.compute_gradients(loss=loss, <list of variables>)
	# # with tf.device('/device:GPU:1'):

	# 	train_step=opt.apply_gradients(gradient_all,global_step=tf.train.get_global_step(), name='train_step')

	merged_summary_op = tf.summary.merge_all()

	return x, y, istrain, realpred, train_step, merged_summary_op

	# return x, y, istrain, realpred, gradient_all
