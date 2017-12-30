import tensorflow as tf
import numpy as np

NUM_CLASSES = 10

def fire_module(x, inp, sp, e11p, e33p, weight_decay):
    weights = []
    bias = []
    with tf.variable_scope("fire"):
        with tf.variable_scope("squeeze"):
            W = tf.get_variable("weights", shape=[1, 1, inp, sp],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
            b = tf.get_variable("bias", shape=[sp])
            s = tf.nn.conv2d(x, W, [1, 1, 1, 1], "VALID") + b
            s = tf.nn.relu(s)
            weights.append(W)
            bias.append(b)
        with tf.variable_scope("e11"):
            W = tf.get_variable("weights", shape=[1, 1, sp, e11p],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
            b = tf.get_variable("bias", shape=[e11p])
            e11 = tf.nn.conv2d(s, W, [1, 1, 1, 1], "VALID") + b
            e11 = tf.nn.relu(e11)
            weights.append(W)
            bias.append(b)
        with tf.variable_scope("e33"):
            W = tf.get_variable("weights", shape=[3, 3, sp, e33p],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
            b = tf.get_variable("bias", shape=[e33p])
            e33 = tf.nn.conv2d(s, W, [1, 1, 1, 1], "SAME") + b
            e33 = tf.nn.relu(e33)
            weights.append(W)
            bias.append(b)
        return tf.concat([e11,e33],3), weights, bias


class SqueezeNet(object):
    def extract_features(self, weight_decay, input=None, reuse=True):
        if input is None:
            input = self.image
        x = input
        layers = []
        weights = []
        bias = []
        with tf.variable_scope('features', reuse=reuse):
            with tf.variable_scope('conv1'):
                W = tf.get_variable("weights", shape=[3, 3, 3, 64],
                                    initializer=tf.contrib.layers.xavier_initializer(),
                                    regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
                b = tf.get_variable("bias", shape=[64])
                x = tf.nn.conv2d(x, W, [1, 2, 2, 1], "VALID")
                x = tf.nn.bias_add(x, b)
                x = tf.nn.relu(x)
                layers.append(x)
                weights.append(W)
                bias.append(b)
                               
            with tf.variable_scope('fire2'):
                x, fire_weight, fire_bias = fire_module(x, 64, 16, 64, 64, weight_decay)
                layers.append(x)
                weights = weights + fire_weight
                bias = bias + fire_bias
                
            with tf.variable_scope('fire3'):
                x, fire_weight, fire_bias = fire_module(x, 128, 16, 64, 64, weight_decay)
                layers.append(x)
                weights = weights + fire_weight
                bias = bias + fire_bias
                
            with tf.variable_scope('maxpool3'):
                x = tf.nn.max_pool(x, [1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
                layers.append(x)
                
            with tf.variable_scope('fire4'):
                x, fire_weight, fire_bias = fire_module(x, 128, 32, 128, 128, weight_decay)
                layers.append(x)
                weights = weights + fire_weight
                bias = bias + fire_bias
                
            with tf.variable_scope('fire5'):
                x, fire_weight, fire_bias = fire_module(x, 256, 32, 128, 128, weight_decay)
                layers.append(x)
                weights = weights + fire_weight
                bias = bias + fire_bias
                
            with tf.variable_scope('maxpool5'):
                x = tf.nn.max_pool(x, [1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
                layers.append(x)
                
            with tf.variable_scope('fire6'):
                x, fire_weight, fire_bias = fire_module(x, 256, 48, 192, 192, weight_decay)
                layers.append(x)
                weights = weights + fire_weight
                bias = bias + fire_bias
                
            with tf.variable_scope('fire7'):
                x, fire_weight, fire_bias = fire_module(x, 384, 48, 192, 192, weight_decay)
                layers.append(x)
                weights = weights + fire_weight
                bias = bias + fire_bias
                
            with tf.variable_scope('fire8'):
                x, fire_weight, fire_bias = fire_module(x, 384, 64, 256, 256, weight_decay)
                layers.append(x)
                weights = weights + fire_weight
                bias = bias + fire_bias
                
            with tf.variable_scope('fire9'):
                x, fire_weight, fire_bias = fire_module(x, 512, 64, 256, 256, weight_decay)
                x = tf.nn.dropout(x, 0.5)
                layers.append(x)
                weights = weights + fire_weight
                bias = bias + fire_bias

            with tf.variable_scope('conv10'):
                W = tf.get_variable("weights", shape=[1, 1, 512, 10],
                                    initializer=tf.truncated_normal_initializer(stddev=0.01),
                                    regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
                b = tf.get_variable("bias", shape=[10])
                x = tf.nn.conv2d(x, W, [1, 1, 1, 1], "VALID")
                x = tf.nn.bias_add(x,b)
                x = tf.nn.relu(x)
                layers.append(x)
                weights.append(W)
                bias.append(b)

            with tf.variable_scope('avgpool10'):
                x = tf.nn.avg_pool(x, [1, 3, 3, 1], strides=[1, 3, 3, 1], padding='VALID')
                layers.append(x)
                
        return layers, weights, bias

    def __init__(self, weight_decay):
        self.image = tf.placeholder('float', shape=[None, None, None, 3], name='input_image')
        self.labels = tf.placeholder('int32', shape=[None], name='labels')

        self.layers = []
        self.weights = []
        self.bias = []

        self.layers, self.weights, self.bias = self.extract_features(weight_decay=weight_decay, input=self.image, reuse=False)
        x = self.layers[-1]
                
        self.classifier = tf.reshape(x, [-1, NUM_CLASSES])

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.labels, NUM_CLASSES),
                                                                           logits=self.classifier)
                                   )


