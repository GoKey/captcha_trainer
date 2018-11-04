#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import tensorflow as tf
from tensorflow.python.training import moving_averages
from config import *


class LSTM(object):

    def __init__(self, mode):
        self.mode = mode
        self.seq_len = tf.placeholder(tf.int32, [None], name="seq_len")
        self.inputs = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, 1], name='input')
        self.labels = tf.sparse_placeholder(tf.int32, name='labels')
        self._extra_train_ops = []
        self.merged_summary = None

    def build_graph(self):
        self._build_model()
        self._build_train_op()
        self.merged_summary = tf.summary.merge_all()

    def _build_model(self):
        with tf.variable_scope('cnn'):
            x = self.inputs
            for i, neu in enumerate(CNN_STRUCTURE):
                with tf.variable_scope('unit-%d' % (i + 1)):

                    x = self._conv2d(x, 'cnn-%d' % (i + 1), CONV_KSIZE[i], FILTERS[i][0], FILTERS[i][1], CONV_STRIDES[i])
                    x = self._batch_norm('bn%d' % (i + 1), x)
                    x = self._leaky_relu(x, LEAKINESS)
                    x = self._max_pool(x, POOL_KSIZE[i], POOL_STRIDES[i])

            _, feature_h, feature_w, _ = x.get_shape().as_list()

            with tf.variable_scope('lstm'):

                x = tf.reshape(x, [-1, OUT_CHANNEL, feature_h * feature_w])

                # lstm_fw_cell = tf.contrib.rnn.LSTMCell(NUM_HIDDEN // 2, state_is_tuple=True)
                # lstm_bw_cell = tf.contrib.rnn.LSTMCell(NUM_HIDDEN // 2, state_is_tuple=True)
                # output, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, x, img_len, dtype=tf.float32)

                cell = tf.contrib.rnn.LSTMCell(NUM_HIDDEN, state_is_tuple=True)
                if self.mode == RunMode.Trains:
                    cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=OUTPUT_KEEP_PROB)

                cell1 = tf.contrib.rnn.LSTMCell(NUM_HIDDEN, state_is_tuple=True)
                if self.mode == RunMode.Trains:
                    cell1 = tf.contrib.rnn.DropoutWrapper(cell=cell1, output_keep_prob=OUTPUT_KEEP_PROB)

                # Stacking rnn cells
                stack = tf.contrib.rnn.MultiRNNCell([cell, cell1], state_is_tuple=True)

                initial_state = stack.zero_state(tf.shape(self.inputs)[0], tf.float32)

                outputs, _ = tf.nn.dynamic_rnn(
                    cell=stack,
                    inputs=x,
                    sequence_length=self.seq_len,
                    initial_state=initial_state,
                    dtype=tf.float32,
                    scope="rnn_output"
                )

                # Reshaping to apply the same weights over the time_steps
                outputs = tf.reshape(outputs, [-1, NUM_HIDDEN])
                with tf.variable_scope('output'):
                    weight_out = tf.get_variable(
                        name='weight',
                        shape=[NUM_HIDDEN, NUM_CLASSES],
                        dtype=tf.float32,
                        # initializer=tf.contrib.layers.xavier_initializer(),
                        initializer=tf.truncated_normal_initializer(stddev=0.1),
                        # initializer=tf.truncated_normal([NUM_HIDDEN, NUM_CLASSES], stddev=0.1),
                    )
                    biases_out = tf.get_variable(
                        name='biases',
                        shape=[NUM_CLASSES],
                        dtype=tf.float32,
                        initializer=tf.constant_initializer()
                    )
                    # [batch_size * max_timesteps, num_classes]
                    output = tf.add(tf.matmul(outputs, weight_out), biases_out)
                    # [batch_size, max_timesteps, num_classes]
                    output = tf.reshape(output, [tf.shape(x)[0], -1, NUM_CLASSES])
                    # [max_timesteps, batch_size, num_classes]
                    predict = tf.transpose(output, (1, 0, 2), "predict")
                    self.predict = predict

    def _build_train_op(self):

        self.global_step = tf.train.get_or_create_global_step()

        # ctc loss function, using forward and backward algorithms and maximum likelihood.
        self.loss = tf.nn.ctc_loss(
            labels=self.labels,
            inputs=self.predict,
            sequence_length=self.seq_len,
        )
        self.cost = tf.reduce_mean(self.loss)
        tf.summary.scalar('cost', self.cost)

        self.lrn_rate = tf.train.exponential_decay(
            TRAINS_LEARNING_RATE,
            self.global_step,
            DECAY_STEPS,
            DECAY_RATE,
            staircase=True
        )
        tf.summary.scalar('learning_rate', self.lrn_rate)

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.lrn_rate,
            beta1=BATE1,
            beta2=BATE2
        ).minimize(
            self.loss,
            global_step=self.global_step
        )

        # Storing adjusted smoothed mean and smoothed variance operations
        train_ops = [self.optimizer] + self._extra_train_ops
        self.train_op = tf.group(*train_ops)

        # Option 2: tf.contrib.ctc.ctc_beam_search_decoder
        # (it's slower but you'll get better results)
        # self.decoded, self.log_prob = tf.nn.ctc_greedy_decoder(
        #     self.predict,
        #     self.seq_len,
        #     merge_repeated=False
        # )

        # Find the optimal path
        self.decoded, self.log_prob = tf.nn.ctc_beam_search_decoder(
            self.predict,
            self.seq_len,
            merge_repeated=False,
        )

        self.dense_decoded = tf.sparse_tensor_to_dense(self.decoded[0], default_value=-1)

    @staticmethod
    def _conv2d(x, name, filter_size, in_channels, out_channels, strides):
        with tf.variable_scope(name):
            kernel = tf.get_variable(
                name='weight',
                shape=[filter_size, filter_size, in_channels, out_channels],
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())

            biases = tf.get_variable(
                name='biases',
                shape=[out_channels],
                dtype=tf.float32,
                initializer=tf.constant_initializer()
            )

            con2d_op = tf.nn.conv2d(x, kernel, [1, strides, strides, 1], padding='SAME')

        return tf.nn.bias_add(con2d_op, biases)

    def _batch_norm(self, name, x):
        with tf.variable_scope(name):

            # Get the last dimension of tensor, the mean after, the variance is this dimension
            params_shape = [x.get_shape()[-1]]

            # Normalized data is the mean value of 0 after the variance is 1,
            # - there is also an adjustment of x = x * gamma + beta
            # This will continue to adjust with training
            beta = tf.get_variable(
                'beta', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable(
                'gamma', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32))

            # When training, constantly adjust the smoothing mean, smoothing the variance
            # In the prediction process, the adjusted smooth variance mean is used for standardization during training.
            if self.mode == RunMode.Trains:
                # Get batch average and variance, size[Last Dimension]
                mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

                # These two names, moving_mean and moving_variance must be equal to both training and prediction
                # - get_variable() can be used to create shared variables
                moving_mean = tf.get_variable(
                    'moving_mean', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32),
                    trainable=False)
                moving_variance = tf.get_variable(
                    'moving_variance', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32),
                    trainable=False)

                self._extra_train_ops.append(moving_averages.assign_moving_average(
                    moving_mean, mean, 0.9))
                self._extra_train_ops.append(moving_averages.assign_moving_average(
                    moving_variance, variance, 0.9))
            else:
                mean = tf.get_variable(
                    'moving_mean', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32),
                    trainable=False)
                variance = tf.get_variable(
                    'moving_variance', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32),
                    trainable=False)

                tf.summary.histogram(mean.op.name, mean)
                tf.summary.histogram(variance.op.name, variance)

            x_bn = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)
            x_bn.set_shape(x.get_shape())

            return x_bn

    # Variant Relu
    # The gradient of the non-negative interval is constant,
    # - which can prevent the gradient from disappearing to some extent.
    @staticmethod
    def _leaky_relu(x, leakiness=0.0):
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    @staticmethod
    def _max_pool(x, ksize, strides):
        return tf.nn.max_pool(
            x,
            ksize=[1, ksize, ksize, 1],
            strides=[1, strides, strides, 1],
            padding='SAME',
            name='max_pool'
        )
