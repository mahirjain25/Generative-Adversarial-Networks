import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plot

def weight(a):
    return tf.Variable(tf.truncated_normal(a, stddev=0.1))

def bias(a):
    return tf.Variable(tf.constant(0.1, shape=a))

def conv_layer(x,filter_size,inp_dim,out_dim):
    return tf.nn.conv2d(x,weight([filter_size,filter_size,inp_dim,out_dim]), strides=[1, 1, 1, 1], padding='SAME')

def deconv_layer(x,filter_size,inp_dim,out_dim):
    return tf.nn.conv2d_transpose(x,weight([filter_size,filter_size,inp_dim,out_dim]), strides=[1, 1, 1, 1], padding='SAME')

def linear(x, unit):
    return tf.layers.dense(inputs=x, units=unit,activation=None)

def maxpool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def sigmoid(x):
    return tf.nn.sigmoid(x)

def relu(x):
    return tf.nn.relu(x)

def tanh(x):
    return tf.nn.tanh(x)

def reshape(x, shape):
    return tf.reshape(x, shape=shape)

def conv_concat(x, y):
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()

    return concat([x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], axis=3)

def concat(x, axis=1):
    return tf.concat(x, axis=axis)

def dropout(x,keep_prob):
    return tf.nn.dropout(x, keep_prob)

def gaussian_noise_layer(x, std=0.15):
    noise = tf.random_normal(shape=tf.shape(x), mean=0.0, stddev=std, dtype=tf.float32)
    return x + noise

def GAP(x):
    return global_avg_pool(x)

def flatten(x):
    return tf.contrib.layers.flatten(x)

def lrelu(x,alpha):
    return tf.maximum(x,alpha*x)

def batch_norm(x, is_training, scope):
    return tf.contrib.layers.batch_norm(x,
                                        decay=0.9,
                                        updates_collections=None,
                                        epsilon=1e-5,
                                        scale=True,
                                        is_training=is_training,
                                        scope=scope)


def display(sample):
    image = sample.reshape([280,280])
    plot.axis('off')
    plot.imshow(image, cmap=matplotlib.cm.binary)
    plot.show()
