import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

class ConvNeuralNet(object):

    def __init__(self, data, label, training=None, height=None, width=None, channel=None, classes=None):

        print("\n** Initialize CNN Layers")

        input_data = tf.reshape(data, [-1, height, width, channel])
        print("Input: "+str(input_data.shape))

        self._conv_1 = self.convolution(inputs=input_data, filters=8, k_size=5, stride=1, padding="same")
        self._maxpool_1 = self.maxpool(inputs=self._conv_1, pool_size=2)

        self._conv_2 = self.convolution(inputs=self._maxpool_1, filters=16, k_size=5, stride=1, padding="same")
        self._maxpool_2 = self.maxpool(inputs=self._conv_2, pool_size=2)

        self._conv_3 = self.convolution(inputs=self._maxpool_2, filters=32, k_size=5, stride=1, padding="same")
        self._maxpool_3 = self.maxpool(inputs=self._conv_3, pool_size=2)

        self._full_conv_1 = self.convolution(inputs=self._maxpool_3, filters=classes, k_size=int(self._maxpool_3.shape[1]), stride=1, padding="valid")
        self._full_conv_2 = self.convolution(inputs=self._full_conv_1, filters=classes, k_size=1, stride=1, padding="same", activation_fn=None)
        self._flatten_layer = self.flatten(inputs=self._full_conv_2)

        self._cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self._flatten_layer, labels=label)
        self._loss = tf.reduce_mean(self._cross_entropy) # Equivalent to np.mean

        """https://www.tensorflow.org/versions/r0.12/api_docs/python/train/decaying_the_learning_rate"""
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 0.0001
        self._learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 10000, 0.96, staircase=True)

        self._train_step = tf.train.AdamOptimizer(self._learning_rate, beta1=0.5).minimize(self._loss)

        self._prediction = tf.contrib.layers.softmax(self._flatten_layer) # Want to prediction Use this!
        correct_pred = tf.equal(tf.argmax(self._flatten_layer, 1), tf.argmax(label, 1))
        self._accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def convolution(self, inputs=None, filters=32, k_size=3, stride=1, padding="same", activation_fn=tf.nn.relu):

        """https://www.tensorflow.org/api_docs/python/tf/layers/conv1d"""

        # initializers.xavier_initializer()
        # tf.contrib.keras.initializers.he_normal()

        conv = tf.contrib.layers.conv2d(
        inputs=inputs,
        num_outputs=filters,
        kernel_size=k_size,
        stride=stride,
        padding=padding,
        data_format=None,
        rate=1,
        activation_fn=activation_fn,
        normalizer_fn=None,
        normalizer_params=None,
        weights_initializer=tf.contrib.keras.initializers.he_normal(),
        weights_regularizer=None,
        biases_initializer=tf.zeros_initializer(),
        biases_regularizer=None,
        reuse=None,
        variables_collections=None,
        outputs_collections=None,
        trainable=True,
        scope=None
        )

        print("Convolution: "+str(conv.shape))
        return conv

    def deconvolution(self, inputs=None, filters=32, k_size=3, stride=1, padding="same"):

        deconv = tf.contrib.layers.conv2d_transpose(
        inputs=inputs,
        num_outputs=filters,
        kernel_size=k_size,
        stride=stride,
        padding=padding,
        data_format=None,
        activation_fn=tf.nn.relu,
        normalizer_fn=None,
        normalizer_params=None,
        weights_initializer=tf.contrib.keras.initializers.he_normal(),
        weights_regularizer=None,
        biases_initializer=tf.zeros_initializer(),
        biases_regularizer=None,
        reuse=None,
        variables_collections=None,
        outputs_collections=None,
        trainable=True,
        scope=None
        )

        print("Deconvolution: "+str(deconv.shape))
        return deconv

    def maxpool(self, inputs=None, pool_size=2):

        """https://www.tensorflow.org/api_docs/python/tf/layers/max_pooling1d"""

        maxp = tf.contrib.layers.max_pool2d(
        inputs=inputs,
        kernel_size=pool_size,
        stride=pool_size,
        padding='VALID',
        outputs_collections=None,
        scope=None
        )

        print("Max Pool: "+str(maxp.shape))
        return maxp

    def avgpool(self, inputs=None, pool_size=2):

        """https://www.tensorflow.org/api_docs/python/tf/layers/average_pooling1d"""

        avg = tf.contrib.layers.avg_pool2d(
        inputs=inputs,
        kernel_size=pool_size,
        stride=pool_size,
        padding='VALID',
        outputs_collections=None,
        scope=None
        )

        print("Average Pool: "+str(avg.shape))
        return avg

    def flatten(self, inputs=None):

        """https://www.tensorflow.org/api_docs/python/tf/contrib/layers/flatten"""

        flat = tf.contrib.layers.flatten(inputs=inputs)

        print("Flatten: "+str(flat.shape))
        return flat

    def fully_connected(self, inputs=None, num_outputs=None, activate_fn=None):

        """https://www.tensorflow.org/api_docs/python/tf/contrib/layers/fully_connected"""

        full_con = tf.contrib.layers.fully_connected(inputs=inputs, num_outputs=num_outputs, activation_fn=activate_fn)

        print("Fully Connected: "+str(full_con.shape))
        return full_con

    def batch_normalization(self, inputs=None):
        batchnorm = tf.contrib.layers.batch_norm(
        inputs=inputs,
        decay=0.999,
        center=True,
        scale=False,
        epsilon=0.001,
        activation_fn=None,
        param_initializers=None,
        param_regularizers=None,
        updates_collections=tf.GraphKeys.UPDATE_OPS,
        is_training=True,
        reuse=None,
        variables_collections=None,
        outputs_collections=None,
        trainable=True,
        batch_weights=None,
        fused=None,
        data_format=DATA_FORMAT_NHWC,
        zero_debias_moving_mean=False,
        scope=None,
        renorm=False,
        renorm_clipping=None,
        renorm_decay=0.99
        )

        return batchnorm

    def dropout(self, inputs=None, ratio=0.5, train=None):

        """https://www.tensorflow.org/api_docs/python/tf/layers/dropout"""

        drop = tf.layers.dropout(
        inputs=inputs,
        rate=ratio,
        noise_shape=None,
        seed=None,
        training=train,
        name=None
        )

        print("Dropout: "+str(ratio))
        return drop

    def residual_block(self, inputs=None, filters=1, k_size=5, stride=1):

        conv_1 = convolution(inputs=inputs, filters=filters, k_size=k_size, stride=stride, padding="same")
        conv_2 = convolution(inputs=conv_1, filters=filters, k_size=k_size, stride=stride, padding="same")
        conv_3 = convolution(inputs=conv_2, filters=filters, k_size=k_size, stride=stride, padding="same")

        resi = convolution(inputs=inputs, filters=filters, k_size=1, stride=stride, padding="same")

        return conv_3+resi
