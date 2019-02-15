# -*- coding: utf-8 -*-
import os

choice = 0
try:
    if os.sys.argv[1] == 'train':
        choice = 1
    elif os.sys.argv[1] == 'valid':
        choice = 2
except:
    print('请提供选项')
    exit(-1)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from kinect_data import KinectDataLoader

lr = 0.00001
training_iters = 100000
batch_size = 128

n_inputs = 63
n_steps = 200
n_hidden_units = 1024
n_classes = 6

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 创建卷积层
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# 2x1池化层
def max_pool_2x1(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME')

# 2x3池化层
def max_pool_2x3(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 3, 1], strides=[1, 2, 3, 1], padding='SAME')

xs = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
ys = tf.placeholder(tf.float32, [None, n_classes])

xs = tf.reshape(xs, [-1, n_steps, n_inputs, 1])

# 卷积层1
W_conv1 = weight_variable([5, 5, 1, 4])  # (128, 200, 63, 4)
b_conv1 = bias_variable([4])
h_conv1 = (conv2d(xs, W_conv1) + b_conv1)

# 池化层1
h_pool1 = max_pool_2x3(h_conv1)  # (128, 100, 21, 4)

# 卷积层2
W_conv2 = weight_variable([5, 5, 4, 8])  # (128, 100, 21, 8)
b_conv2 = bias_variable([8])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

# 池化层2
h_pool2 = max_pool_2x3(h_conv2)  # (128, 50, 7, 8)

# 卷积层3
W_conv3 = weight_variable([5, 5, 8, 16])  # (128, 50, 21, 16)
b_conv3 = bias_variable([16])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

# 池化层3
h_pool3 = max_pool_2x1(h_conv3)  # (128, 25, 7, 16)


# nn
W_fc1 = weight_variable([25*7*16, n_hidden_units])
b_fc1 = bias_variable([n_hidden_units])
h_pool3_flat = tf.reshape(h_pool3, [-1, 25*7*16])
h_fc1 = tf.matmul(h_pool3_flat, W_fc1) + b_fc1
h_fc1 = tf.nn.dropout(h_fc1, keep_prob=0.95)

W_fc2 = weight_variable([n_hidden_units, n_classes])
b_fc2 = bias_variable([n_classes])
pred = tf.matmul(h_fc1, W_fc2) + b_fc2

# cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=ys))

# train
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

init = tf.initialize_all_variables()
saver = tf.train.Saver(max_to_keep=1)
max_accuracy = 0

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('steps')
ax.set_ylabel('cost')
plt.ion()
plt.show()

graph_steps = []
graph_cost = []

with tf.Session() as sess:
    if choice == 1:
        kinect_data_loader = KinectDataLoader()
        sess.run(init)
        # model_file = tf.train.latest_checkpoint('ckpt_cnn/')
        # saver.restore(sess, model_file)
        step = 0
        accuracy = 0
        while True:
            batch_xs, batch_ys = kinect_data_loader.next_batch(batch_size)
            batch_xs = batch_xs.reshape([batch_size, n_steps, 63, 1])
            _, loss = sess.run([train_op, cost], feed_dict={
                xs: batch_xs,
                ys: batch_ys
            })
            graph_steps.append(step)
            graph_cost.append(loss)
            ax.plot(graph_steps, graph_cost)
            plt.pause(0.1)
            if step % 5 == 0:
                total = 0
                correct = 0
                test_xs, test_ys = kinect_data_loader.next_batch(batch_size)
                test_xs = test_xs.reshape([batch_size, n_steps, 63, 1])
                test_output_y = sess.run(pred, feed_dict={
                    xs: test_xs,
                    ys: test_ys
                })
                y_res = tf.argmax(test_ys, 1)
                output_res = tf.argmax(test_output_y, 1)
                for p, q in zip(sess.run(y_res), sess.run(output_res)):
                    total += 1
                    correct += (p == q)
                accuracy = correct / total
                print("Accuracy on training set: " + str(accuracy))
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                print('Max Accuracy has reached: ' + str(max_accuracy))
            if accuracy >= max_accuracy:
                saver.save(sess, 'ckpt_cnn/trained.ckpt', global_step=step)
            step += 1

    elif choice == 2:
        kinect_data_loader = KinectDataLoader('./Kinect_test/')
        model_file = tf.train.latest_checkpoint('ckpt_cnn/')
        saver.restore(sess, model_file)
        total = 0
        correct = 0
        for test_xs, test_ys in kinect_data_loader.all_batches(batch_size):
            test_xs = test_xs.reshape([batch_size, n_steps, 63, 1])
            test_output_y = sess.run(pred, feed_dict={
                xs: test_xs,
                ys: test_ys
            })
            y_res = tf.argmax(test_ys, 1)
            output_res = tf.argmax(test_output_y, 1)
            for p, q in zip(sess.run(y_res), sess.run(output_res)):
                total += 1
                correct += (p == q)
        print("Accuracy on test set: " + str(correct / total))