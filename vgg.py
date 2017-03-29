from datetime import datetime
import math
import time
import numpy as np
import tensorflow.python.platform
import tensorflow as tf
import vgg_input
import vgg_flop_input
import pickle

IMAGE_SIZE = 39

def init():
	global dico
	with open('dictLabels.pkl','rb') as f:
		dico = pickle.load(f)
	
def conv_op(input_op, name, kw, kh, n_out, dw, dh, reuse):
	n_in = input_op.get_shape()[-1].value
	with tf.variable_scope(name,reuse):
		kernel = tf.get_variable('w',shape=[kh, kw, n_in, n_out],dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer_conv2d())
		conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw,1), padding='SAME')
		#bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
		#biases = tf.Variable(bias_init_val, trainable=True, name='b')
		biases = tf.get_variable(name='b', shape=[n_out],dtype=tf.float32)
		z = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
		activation = tf.nn.relu(z, name=name)
		return activation
     

def fc_op(input_op, name, n_out, reuse, reluBool=True):
	n_in = input_op.get_shape()[-1].value
	with tf.variable_scope(name,reuse):
		kernel = tf.get_variable('w',shape=[n_in, n_out],dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
		#biases = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), name='b')
		biases = tf.get_variable(name='b', shape=[n_out],dtype=tf.float32)
		if reluBool:
			activation = tf.nn.relu_layer(input_op, kernel, biases, name=name)
		else:
			activation = tf.add(tf.matmul(input_op, kernel),biases)
		return activation

def mpool_op(input_op, name, kh, kw, dh, dw):
    return tf.nn.max_pool(input_op,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding='SAME',
                          name=name)

def loss_op(logits, labels, batch_size):
    loss = tf.nn.l2_loss(logits-labels)
    return loss



def evaluate(logits, labels, accuracy):
    # Return probabilities of yau, pitch, roll being less than 5 degrees
    accuracy= accuracy/180
    yauDeltas = tf.abs(logits[:,0]-labels[:,0])
    yauAcc = tf.less_equal(yauDeltas, accuracy)
    yauAcc = tf.reduce_sum(tf.cast(yauAcc, tf.float32))
    yauAcc = yauAcc/tf.to_float(labels.get_shape()[0])

    pitchDeltas = tf.abs(logits[:,1]-labels[:,1])
    pitchAcc = tf.less_equal(pitchDeltas, accuracy)
    pitchAcc = tf.reduce_sum(tf.cast(pitchAcc, tf.float32))
    pitchAcc = pitchAcc/tf.to_float(labels.get_shape()[0])

    rollDeltas = tf.abs(logits[:,2]-labels[:,2])
    rollAcc = tf.less_equal(rollDeltas, accuracy)
    rollAcc = tf.reduce_sum(tf.cast(rollAcc, tf.float32))
    rollAcc = rollAcc/tf.to_float(labels.get_shape()[0])
    return yauAcc, pitchAcc, rollAcc


def inference_vgg(input_op, reuse, training=False):

    dropout_keep_prob = 0.5 if training else 1.0
   

    conv1_1 = conv_op(input_op, name="conv1_1", kh=3, kw=3, n_out=32, dh=1, dw=1, reuse=reuse)
    conv1_2 = conv_op(conv1_1,  name="conv1_2", kh=3, kw=3, n_out=32, dh=1, dw=1,reuse=reuse)
    pool1 = mpool_op(conv1_2,   name="pool1",   kh=2, kw=2, dw=2, dh=2)


    conv2_1 = conv_op(pool1,    name="conv2_1", kh=3, kw=3, n_out=64, dh=1, dw=1,reuse=reuse)
    conv2_2 = conv_op(conv2_1,  name="conv2_2", kh=3, kw=3, n_out=32, dh=1, dw=1,reuse=reuse)
    pool2 = mpool_op(conv2_2,   name="pool2",   kh=2, kw=2, dh=2, dw=2)

    conv3_1 = conv_op(pool2,    name="conv3_1", kh=3, kw=3, n_out=64, dh=1, dw=1,reuse=reuse)
    conv3_2 = conv_op(conv3_1,  name="conv3_2", kh=3, kw=3, n_out=64, dh=1, dw=1,reuse=reuse)
    pool3 = mpool_op(conv3_2,   name="pool3",   kh=2, kw=2, dh=2, dw=2)



    # flatten
    shp = pool3.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    resh1 = tf.reshape(pool3, [-1, flattened_shape], name="resh1")

    # fully connected
    fc6 = fc_op(resh1, name="fc6", n_out=64,reuse=reuse)
    fc6_drop = tf.nn.dropout(fc6, dropout_keep_prob, name="fc6_drop")


    fc8 = fc_op(fc6_drop, name="fc8", n_out=3,reuse=reuse, reluBool = False)
    return fc8
	
def inference_fc6(input_op, reuse):

    conv1_1 = conv_op(input_op, name="conv1_1", kh=3, kw=3, n_out=32, dh=1, dw=1, reuse=reuse)
    conv1_2 = conv_op(conv1_1,  name="conv1_2", kh=3, kw=3, n_out=32, dh=1, dw=1,reuse=reuse)
    pool1 = mpool_op(conv1_2,   name="pool1",   kh=2, kw=2, dw=2, dh=2)


    conv2_1 = conv_op(pool1,    name="conv2_1", kh=3, kw=3, n_out=64, dh=1, dw=1,reuse=reuse)
    conv2_2 = conv_op(conv2_1,  name="conv2_2", kh=3, kw=3, n_out=32, dh=1, dw=1,reuse=reuse)
    pool2 = mpool_op(conv2_2,   name="pool2",   kh=2, kw=2, dh=2, dw=2)

    conv3_1 = conv_op(pool2,    name="conv3_1", kh=3, kw=3, n_out=64, dh=1, dw=1,reuse=reuse)
    conv3_2 = conv_op(conv3_1,  name="conv3_2", kh=3, kw=3, n_out=64, dh=1, dw=1,reuse=reuse)
    pool3 = mpool_op(conv3_2,   name="pool3",   kh=2, kw=2, dh=2, dw=2)



    # flatten
    shp = pool3.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    resh1 = tf.reshape(pool3, [-1, flattened_shape], name="resh1")

    # fully connected
    fc6 = fc_op(resh1, name="fc6", n_out=64,reuse=reuse)
    return fc6
	

def labeled_inputs(data, batch_size):
  """Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1] size.
    labels: Labels. 1D tensor of [batch_size, 3] size.
  """
  if(data=='training'):
	data_dir = "trainingImagesPaths.txt"
  elif(data=='test'):
	data_dir = "validationImagesPaths.txt"
  else:
	raise Exception('Please choose training or test as first argument')
  images, paths = vgg_input.labeled_inputs(data_dir=data_dir,
                                                  batch_size=batch_size)

  labels = tf.py_func(getLabels,[paths],[tf.float32])
  labels = tf.convert_to_tensor(labels, dtype = tf.float32)
  labels = tf.reshape(labels,[batch_size,3])
  return images, labels

def getLabels(paths):
	batch_size=len(paths)
	labelsEval = np.zeros([batch_size,3], dtype=float)
	for i in range(0,batch_size):
		labelsEval[i,:]=dico[paths[i]]
        labelsEval = np.reshape(labelsEval,(batch_size,3))
        labelsEval = labelsEval.astype(np.float32)
	return labelsEval

def flop_inputs(data, batch_size):
  """Construct flop input.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1] size.
  """
  if(data=='training'):
	#data_dir = "trainingImagesFlopPaths.txt"
	data_dir = "validationImagesPaths.txt"
  elif(data=='test'):
	data_dir = "validationImagesFlopPaths.txt"
  else:
	raise Exception('Please choose training or test as first argument')
  images = vgg_flop_input.inputs(data_dir=data_dir,
                                                  batch_size=batch_size)

  return images
  
def loss_flop(logitsFlop, logitsFlop2,batch_size):
	pitchs1 = tf.slice(logitsFlop, [0,1],[batch_size,1])
	pitchs2 = tf.slice(logitsFlop2, [0,1],[batch_size,1])
	yaws1 = tf.slice(logitsFlop, [0,0],[batch_size,1])
	yaws2 = tf.slice(logitsFlop2, [0,0],[batch_size,1])
	rolls1 = tf.slice(logitsFlop, [0,2],[batch_size,1])
	rolls2 = tf.slice(logitsFlop2, [0,2],[batch_size,1])
	loss = tf.nn.l2_loss(pitchs1-pitchs2)+tf.nn.l2_loss(yaws1+yaws2)+tf.nn.l2_loss(rolls1+rolls2)
	return loss
	
def loss_flop0Pen(logitsFlop, logitsFlop2,batch_size):
	pitchs1 = tf.slice(logitsFlop, [0,1],[batch_size,1])
	pitchs2 = tf.slice(logitsFlop2, [0,1],[batch_size,1])
	yaws1 = tf.slice(logitsFlop, [0,0],[batch_size,1])
	yaws2 = tf.slice(logitsFlop2, [0,0],[batch_size,1])
	rolls1 = tf.slice(logitsFlop, [0,2],[batch_size,1])
	rolls2 = tf.slice(logitsFlop2, [0,2],[batch_size,1])
	loss = tf.nn.l2_loss(pitchs1-pitchs2)/(tf.nn.l2_loss(pitchs1)+tf.nn.l2_loss(pitchs2))+tf.nn.l2_loss(yaws1+yaws2)/(tf.nn.l2_loss(yaws1)+tf.nn.l2_loss(yaws2))+tf.nn.l2_loss(rolls1+rolls2)/(tf.nn.l2_loss(rolls1)+tf.nn.l2_loss(rolls2))
	return loss
	
def loss_flopMaxVar(logitsFlop, logitsFlop2,batch_size):
	pitchs1 = tf.slice(logitsFlop, [0,1],[batch_size,1])
	pitchs2 = tf.slice(logitsFlop2, [0,1],[batch_size,1])
	yaws1 = tf.slice(logitsFlop, [0,0],[batch_size,1])
	yaws2 = tf.slice(logitsFlop2, [0,0],[batch_size,1])
	rolls1 = tf.slice(logitsFlop, [0,2],[batch_size,1])
	rolls2 = tf.slice(logitsFlop2, [0,2],[batch_size,1])
	(mean,var) = tf.nn.moments(logitsFlop-logitsFlop2, [0])
	varPitch = var[1]
	loss = (1./100.)*tf.nn.l2_loss(pitchs1-pitchs2)/varPitch+tf.nn.l2_loss(yaws1+yaws2)/(tf.nn.l2_loss(yaws1)+tf.nn.l2_loss(yaws2))+tf.nn.l2_loss(rolls1+rolls2)/(tf.nn.l2_loss(rolls1)+tf.nn.l2_loss(rolls2))
	return loss
	
def noise_image(image):
	""" image -> image
		given an image, apply a noise """
	offset_height =  tf.random_uniform([], minval=103, maxval=137, dtype=tf.int32)
	offset_width =  tf.random_uniform([], minval=75, maxval=109, dtype=tf.int32)
	im_size = tf.random_uniform([], minval=151, maxval=185, dtype=tf.int32)
	image_output = tf.image.crop_to_bounding_box(image, offset_height, offset_width, im_size, im_size)
	image_output = tf.image.resize_images(image_output, IMAGE_SIZE, IMAGE_SIZE)
	image_output = tf.slice(image_output, [0,0,0], [IMAGE_SIZE, IMAGE_SIZE, 1])
	return image_output
	
def noise_batch(images):
	""" images -> images 
		given a batch of images, apply a noise to every single image of the batch"""
	return(tf.map_fn(noise_image, images))



	
