# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from kinect_data import KinectDataLoader

kinect_data_loader = KinectDataLoader()

lr = 0.0001
training_iters = 100000
batch_size = 256

n_inputs = 63
n_steps = 200
n_hidden_units = 128
n_classes = 5

x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

weights = {
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units,])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes,]))
}

def RNN(X, weights, biases):
    X = tf.reshape(X, [-1, n_inputs])
    X_in = tf.matmul(X, weights['in']) + biases['in']
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    _, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state, time_major=False)

    results = tf.matmul(states[1], weights['out']) + biases['out']
    return results

pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = kinect_data_loader.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, 63])
        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys
        })
        if step % 20 == 0:
            test_xs, test_ys = kinect_data_loader.next_batch(batch_size)
            test_xs = test_xs.reshape([batch_size, n_steps, 63])
            test_output_y = sess.run(pred, feed_dict={
                x: test_xs,
                y: test_ys
            })
            y_res = tf.argmax(test_ys, 1)
            output_res = tf.argmax(test_output_y, 1)
            hit = 0
            for p, q in zip(sess.run(y_res), sess.run(output_res)):
                hit += (p == q)
            print("Accuracy: " + str(hit / len(sess.run(y_res))))
        step += 1