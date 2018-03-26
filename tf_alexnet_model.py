import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from img_data_utils import *
import math
import time


class AlexNet(object):
    """AlexNet
    """
    def __init__(self, reg_strength=1e-3):
        tf.reset_default_graph()
        self.reg_strength = reg_strength

        # Define placeholders
        self.X = tf.placeholder(tf.float32, [None, 227, 227, 3])
        self.y = tf.placeholder(tf.int64, [None])
        self.is_training = tf.placeholder(tf.bool)
        self.num_classes = 1000

        layers = []
        layers.append(tf.layers.conv2d(inputs=self.X,
                                       filters=96,
                                       kernel_size=[11, 11],
                                       strides=(4, 4),
                                       padding='VALID',
                                       activation=tf.nn.relu,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                       activity_regularizer=tf.contrib.layers.l2_regularizer(reg_strength)))
        layers.append(tf.layers.max_pooling2d(inputs=layers[-1], pool_size=[3, 3], strides=2))
        layers.append(tf.layers.batch_normalization(inputs=layers[-1], axis=3, training=self.is_training))
        layers.append(tf.layers.conv2d(inputs=layers[-1],
                                       filters=256,
                                       kernel_size=[5, 5],
                                       strides=(1, 1),
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                       activity_regularizer=tf.contrib.layers.l2_regularizer(reg_strength)))
        layers.append(tf.layers.max_pooling2d(inputs=layers[-1], pool_size=[3, 3], strides=2))
        layers.append(tf.layers.batch_normalization(inputs=layers[-1], axis=3, training=self.is_training))
        layers.append(tf.layers.conv2d(inputs=layers[-1],
                                       filters=384,
                                       kernel_size=[3, 3],
                                       strides=(1, 1),
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                       activity_regularizer=tf.contrib.layers.l2_regularizer(reg_strength)))
        layers.append(tf.layers.conv2d(inputs=layers[-1],
                                       filters=384,
                                       kernel_size=[3, 3],
                                       strides=(1, 1),
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                       activity_regularizer=tf.contrib.layers.l2_regularizer(reg_strength)))
        layers.append(tf.layers.conv2d(inputs=layers[-1],
                                       filters=256,
                                       kernel_size=[3, 3],
                                       strides=(1, 1),
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                       activity_regularizer=tf.contrib.layers.l2_regularizer(reg_strength)))
        layers.append(tf.layers.max_pooling2d(inputs=layers[-1], pool_size=[3, 3], strides=2))
        layers.append(tf.layers.dense(inputs=tf.reshape(layers[-1], [-1, 6*6*256]), units=4096))
        layers.append(tf.layers.dense(inputs=layers[-1], units=4096))
        layers.append(tf.layers.dense(inputs=layers[-1], units=1000))

        print 'Printing layer dimensions'
        self.layers = layers
        for layer in self.layers:
            print layer.shape


def main():
    model = AlexNet()


if __name__ == '__main__':
    main()
