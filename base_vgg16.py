import inspect
import os

import numpy as np
import tensorflow as tf
import time
from network_util import *

VGG_MEAN = [103.939, 116.779, 123.68]

class Vgg16:
    def __init__(self, vgg16_npy_path=None):
        if vgg16_npy_path is None:
            path = inspect.getfile(Vgg16)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg16.npy")
            vgg16_npy_path = path
            print(path)

        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        print("npy file loaded")

    def build_model(self, bgr, from_npy=True):
        """
        load variable from npy to build the VGG
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """
        start_time = time.time()
        print("build model started")
        if from_npy:
            self.conv1_1 = self.conv_layer(bgr, "conv1_1", training=False)
            self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2", training=False)
            self.pool1 = self.max_pool(self.conv1_2, 'pool1')

            self.conv2_1 = self.conv_layer(self.pool1, "conv2_1", training=False)
            self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2", training=False)
            self.pool2 = self.max_pool(self.conv2_2, 'pool2')

            self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
            self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
            self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
            self.pool3 = self.max_pool(self.conv3_3, 'pool3')

            self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
            self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
            self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")

            self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
            self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
            self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")

            self.data_dict = None
        else:
            self.conv1_1 = convLayer(bgr, 3, 64, 3, 1, activation=tf.nn.relu, name="conv1_1")
            self.conv1_2 = convLayer(self.conv1_1, 64, 64, 3, 1, activation=tf.nn.relu, name="conv1_2")
            self.pool1 = maxpool2d(self.conv1_2, kernel=2, stride=2, name="pool1")

            self.conv2_1 = convLayer(self.pool1, 64, 128, 3, 1, activation=tf.nn.relu, name="conv2_1")
            self.conv2_2 = convLayer(self.conv2_1, 128, 128, 3, 1, activation=tf.nn.relu, name="conv2_2")
            self.pool2 = maxpool2d(self.conv2_2, kernel=2, stride=2, name="pool2")

            self.conv3_1 = convLayer(self.pool2, 128, 256, 3, 1, activation=tf.nn.relu, name="conv3_1")
            self.conv3_2 = convLayer(self.conv3_1, 256, 256, 3, 1, activation=tf.nn.relu, name="conv3_2")
            self.conv3_3 = convLayer(self.conv3_2, 256, 256, 3, 1, activation=tf.nn.relu, name="conv3_3")
            self.pool3 = maxpool2d(self.conv3_3, kernel=2, stride=2, name="pool3")

            self.conv4_1 = convLayer(self.pool2, 256, 512, 3, 1, activation=tf.nn.relu, name="conv4_1")
            self.conv4_2 = convLayer(self.conv4_1, 512, 512, 3, 1, activation=tf.nn.relu, name="conv4_2")
            self.conv4_3 = convLayer(self.conv4_2, 512, 512, 3, 1, activation=tf.nn.relu, name="conv4_3")
            self.pool4 = maxpool2d(self.conv4_3, kernel=2, stride=2, name="pool4")

            self.conv5_1 = convLayer(self.pool2, 512, 512, 3, 1, activation=tf.nn.relu, name="conv5_1")
            self.conv5_2 = convLayer(self.conv5_1, 512, 512, 3, 1, activation=tf.nn.relu, name="conv5_2")
            self.conv5_3 = convLayer(self.conv5_2, 512, 512, 3, 1, activation=tf.nn.relu, name="conv5_3")
        print(("build model finished: %ds" % (time.time() - start_time)))

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name, training=True):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name, training=training)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name, training=training)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name, training=True):
        return tf.Variable(self.data_dict[name][0], name="filter", trainable=training)

    def get_bias(self, name, training=True):
        return tf.Variable(self.data_dict[name][1], name="biases", trainable=training)

    def get_fc_weight(self, name):
        return tf.Variable(self.data_dict[name][0], name="weights")
