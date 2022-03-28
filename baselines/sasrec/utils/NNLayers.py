import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
import numpy as np

paramId = 0
biasDefault = False
params = {}
regParams = {}
ita = 0.2
leaky = 0.1

def sparse_dropout(self, A, mask):
    ret = tf.sparse_retain(A, mask)
    return ret

def getParamId():
	global paramId
	paramId += 1
	return paramId

def setIta(ITA):
	ita = ITA

def setBiasDefault(val):
	global biasDefault
	biasDefault = val

def getParam(name):
	return params[name]

def addReg(name, param):
	global regParams
	if name not in regParams:
		regParams[name] = param
	else:
		print('ERROR: Parameter already exists')

def addParam(name, param):
	global params
	if name not in params:
		params[name] = param

def defineRandomNameParam(shape, dtype=tf.float32, reg=False, initializer='xavier', trainable=True):
	name = 'defaultParamName%d'%getParamId()
	return defineParam(name, shape, dtype, reg, initializer, trainable)

def defineParam(name, shape, dtype=tf.float32, reg=False, initializer='xavier', trainable=True):
	global params
	global regParams
	assert name not in params, 'name %s already exists' % name
	if initializer == 'xavier':
		ret = tf.get_variable(name=name, dtype=dtype, shape=shape,
			initializer=xavier_initializer(dtype=tf.float32),
			trainable=trainable)
	elif initializer == 'trunc_normal':
		ret = tf.get_variable(name=name, initializer=tf.random.truncated_normal(shape=[int(shape[0]), shape[1]], mean=0.0, stddev=0.03, dtype=dtype))
	elif initializer == 'zeros':
		ret = tf.get_variable(name=name, dtype=dtype,
			initializer=tf.zeros(shape=shape, dtype=tf.float32),
			trainable=trainable)
	elif initializer == 'ones':
		ret = tf.get_variable(name=name, dtype=dtype, initializer=tf.ones(shape=shape, dtype=tf.float32), trainable=trainable)
	elif not isinstance(initializer, str):
		ret = tf.get_variable(name=name, dtype=dtype,
			initializer=initializer, trainable=trainable)
	else:
		print('ERROR: Unrecognized initializer')
		exit()
	params[name] = ret
	if reg:
		regParams[name] = ret
	return ret

def getOrDefineParam(name, shape, dtype=tf.float32, reg=False, initializer='xavier', trainable=True, reuse=False):
	global params
	global regParams
	if name in params:
		assert reuse, 'Reusing Param %s Not Specified' % name
		if reg and name not in regParams:
			regParams[name] = params[name]
		return params[name]
	return defineParam(name, shape, dtype, reg, initializer, trainable)

def BN(inp, name=None):
	global ita
	dim = inp.get_shape()[1]
	name = 'defaultParamName%d'%getParamId()
	scale = tf.Variable(tf.ones([dim]))
	shift = tf.Variable(tf.zeros([dim]))
	fcMean, fcVar = tf.nn.moments(inp, axes=[0])
	ema = tf.train.ExponentialMovingAverage(decay=0.5)
	emaApplyOp = ema.apply([fcMean, fcVar])
	with tf.control_dependencies([emaApplyOp]):
		mean = tf.identity(fcMean)
		var = tf.identity(fcVar)
	ret = tf.nn.batch_normalization(inp, mean, var, shift,
		scale, 1e-8)
	return ret

def FC(inp, outDim=None, name=None, bias=False, activation=None, reg=False, useBN=False, dropout=None, initializer='xavier', reuse=False, bias_reg=False, biasInitializer='zeros'):
	global params
	global regParams
	global leaky
	inDim = inp.get_shape()[-1]
	temName = name if name!=None else 'defaultParamName%d'%getParamId()
	W = getOrDefineParam(temName, [inDim, outDim], reg=reg, initializer=initializer, reuse=reuse)
	if dropout != None:
		ret = tf.nn.dropout(inp, rate=dropout) @ W
	else:
		ret = inp @ W
	if bias:
		ret = Bias(ret, name=name, reuse=reuse, reg=bias_reg, initializer=biasInitializer)
	if useBN:
		ret = BN(ret)
	if activation != None:
		ret = activate(ret, activation)
	return ret

def Bias(data, name=None, reg=False, reuse=False, initializer='zeros'):
	inDim = data.get_shape()[-1]
	temName = name if name!=None else 'defaultParamName%d'%getParamId()
	temBiasName = temName + 'Bias'
	bias = getOrDefineParam(temBiasName, inDim, reg=False, initializer=initializer, reuse=reuse)
	if reg:
		regParams[temBiasName] = bias
	return data + bias

def ActivateHelp(data, method):
	if method == 'relu':
		ret = tf.nn.relu(data)
	elif method == 'sigmoid':
		ret = tf.nn.sigmoid(data)
	elif method == 'tanh':
		ret = tf.nn.tanh(data)
	elif method == 'softmax':
		ret = tf.nn.softmax(data, axis=-1)
	elif method == 'leakyRelu':
		ret = tf.maximum(leaky*data, data)
	elif method == 'twoWayLeakyRelu6':
		temMask = tf.to_float(tf.greater(data, 6.0))
		ret = temMask * (6 + leaky * (data - 6)) + (1 - temMask) * tf.maximum(leaky * data, data)
	elif method == '-1relu':
		ret = tf.maximum(-1.0, data)
	elif method == 'relu6':
		ret = tf.maximum(0.0, tf.minimum(6.0, data))
	elif method == 'relu3':
		ret = tf.maximum(0.0, tf.minimum(3.0, data))
	else:
		raise Exception('Error Activation Function')
	return ret

def activate(data, method, useBN=False):
	global leaky
	if useBN: ret = BN(data)
	else: ret = data
	ret = ActivateHelp(ret, method)
	return ret

def regularization(names=None, method='L2'):
	ret = 0
	if method == 'L1':
		if names != None:
			for name in names:
				ret += tf.reduce_sum(tf.abs(getParam(name)))
		else:
			for name in regParams:
				ret += tf.reduce_sum(tf.abs(regParams[name]))
	elif method == 'L2':
		if names != None:
			for name in names:
				ret += tf.reduce_sum(tf.square(getParam(name)))
		else:
			for name in regParams:
				ret += tf.reduce_sum(tf.square(regParams[name]))
	return ret

def Dropout(data, rate):
	if rate == None: return data
	else: return tf.nn.dropout(data, rate=rate)

def self_attention(X, his, dim, num_heads, num_units=None, causality=False, dropout=0, is_training=False):
    '''
    X: (?, his, dim)
    his: length of history interactions
    dim: latent dimension
    '''

    if num_units == None: num_units = dim

    Q = defineRandomNameParam([dim, num_units], reg=True)
    K = defineRandomNameParam([dim, num_units], reg=True)
    V = defineRandomNameParam([dim, num_units], reg=True)

    q = X @ Q # (?, his, num_units)
    k = X @ K # (?, his, num_units)
    v = X @ V # (?, his, num_units)

    q = tf.concat(tf.split(q, num_heads, axis=-1), axis=0) # (h * ?, his, dim / h) 
    k = tf.concat(tf.split(k, num_heads, axis=-1), axis=0) # (h * ?, his, dim / h) 
    v = tf.concat(tf.split(v, num_heads, axis=-1), axis=0) # (h * ?, his, dim / h) 

    att = q @ tf.linalg.matrix_transpose(k) / tf.math.sqrt(dim / num_heads) # (h * ?, his, his)
    print('======shapes======', q.shape, k.shape, v.shape, att.shape)
    if causality:
        diag = tf.ones_like(att[0]) # (his, his)
        tril = tf.linalg.LinearOperatorLowerTriangular(diag).to_dense() # (his, his)
        mask = tf.tile(tf.expand_dims(tril, 0), [tf.shape(att)[0], 1, 1]) # (h * ?, his, his)
        padding = tf.ones_like(mask) * (-2e32 + 1)
        att = tf.where(tf.equal(mask, 0), padding, att) # (h * ?, his, his)

    att = tf.nn.softmax(att, axis = -1) # (h * ?, his, his)
    att = tf.layers.dropout(att, rate=dropout, training=is_training)

    Y = att @ v # (h * ?, his, dim / h)
    Y = tf.concat(tf.split(Y, num_heads, axis=0), axis=2) # (?, his, dim)

    Y = Y + X

    return Y

def layer_norm(X):
    '''
    X: (?, ...)
    '''
    inputs_shape = X.get_shape()
    params_shape = inputs_shape[-1:]

    mean, variance = tf.nn.moments(X, [-1], keep_dims=True)
    beta= tf.Variable(tf.zeros(params_shape))
    gamma = tf.Variable(tf.ones(params_shape))
    normalized = (X - mean) / ( (variance + 1e-8) ** (.5) )
    Y = gamma * normalized + beta

    return Y
