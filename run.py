import os, argparse

import numpy as np
import tensorflow as tf

import source.neuralnet as nn
import source.sub_procedure as sproc

from tensorflow.examples.tutorials.mnist import input_data

def main():

    dataset = input_data.read_data_sets('MNIST_data', one_hot=True)

    sess = tf.InteractiveSession()

    data = tf.placeholder(tf.float32)
    label = tf.placeholder(tf.float32)
    training = tf.placeholder(tf.bool)

    convnet = nn.ConvNeuralNet(data=data, label=label, training=training, height=28, width=28, channel=1, classes=10)

    sess.run(tf.global_variables_initializer())

    sproc.train(dataset=dataset, data=data, label=label, training=training, model=convnet, sess=sess, steps=FLAGS.steps, batch=FLAGS.batch)

    sproc.heatmap(dataset=dataset, data=data, label=label, training=training, model=convnet, sess=sess)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=50, help='Default: 10. Batches per iteration, the number of data to be training and testing.')
    parser.add_argument('--steps', type=int, default=1000, help='Default: 1000')
    FLAGS, unparsed = parser.parse_known_args()

    main()
