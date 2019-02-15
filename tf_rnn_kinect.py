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

lr = 0.0001  # 学习率
training_iters = 100000  # epoch
batch_size = 128

n_inputs = 63  # 21个关键点的x,y,z坐标, 构成63维向量
n_steps = 200  # 每个动作强制200帧
n_hidden_units = 128  # 隐藏层神经元个数
n_classes = 6  # 动作分类(右拳、左拳、右踢、左踢、防守、静止)

x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])  # 输入层
y = tf.placeholder(tf.float32, [None, n_classes])  # 输出层

weights = {
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),  # 输入层-隐层 权重
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes])) # 隐层-输出层 权重
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units,])),  # 隐层 bias
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes,]))  # 输出层 bias
}


def RNN(X, weights, biases):
    # 为了方便计算隐层输入
    X = tf.reshape(X, [-1, n_inputs])  # 将 shape=(128批, 200帧, 63) 的动作序列转换为 shape=(128*200, 63)
    
    # linear activated hidden layer
    X_in = tf.matmul(X, weights['in']) + biases['in']  # 计算出隐层输入
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])  # 将隐层输入转换为 shape=(128批, 200帧, 128隐层神经元数)
    X_in = tf.nn.dropout(X_in, keep_prob=0.9)

    # lstm单元
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state, time_major=False)

    results = tf.matmul(states[1], weights['out']) + biases['out']  # shape=(128批, 5种动作)
    return results

pred = RNN(x, weights, biases)  # 预测
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))  # 代价
train_op = tf.train.AdamOptimizer(lr).minimize(cost)  # 训练方法

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
            batch_xs, batch_ys = kinect_data_loader.next_batch(batch_size)  # 读进来的数据 shape=(128 batch_size, 200帧, 21个点, 3 xyz)
            batch_xs = batch_xs.reshape([batch_size, n_steps, 63])  # 把上述 (128, 200, 21, 3) 转为 (128, 200, 63)
            sess.run([train_op], feed_dict={
                x: batch_xs,
                y: batch_ys
            })
            graph_steps.append(step)
            graph_cost.append(sess.run(cost, feed_dict={
                x: batch_xs,
                y: batch_ys
            }))
            ax.plot(graph_steps, graph_cost)
            plt.pause(0.1)
            # 每5个step用一个新的batch测试精确度
            if step % 5 == 0:
                total = 0
                correct = 0
                test_xs, test_ys = kinect_data_loader.next_batch(batch_size)
                test_xs = test_xs.reshape([batch_size, n_steps, 63])
                test_output_y = sess.run(pred, feed_dict={
                    x: test_xs,
                    y: test_ys
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
                saver.save(sess, 'ckpt_rnn/trained.ckpt', global_step=step)
            step += 1

    elif choice == 2:
        kinect_data_loader = KinectDataLoader('./Kinect_test/')
        model_file = tf.train.latest_checkpoint('ckpt_rnn/')
        saver.restore(sess, model_file)
        total = 0
        correct = 0
        for test_xs, test_ys in kinect_data_loader.all_batches(batch_size):
            test_xs = test_xs.reshape([batch_size, n_steps, 63])
            test_output_y = sess.run(pred, feed_dict={
                x: test_xs,
                y: test_ys
            })
            y_res = tf.argmax(test_ys, 1)
            output_res = tf.argmax(test_output_y, 1)
            for p, q in zip(sess.run(y_res), sess.run(output_res)):
                total += 1
                correct += (p == q)
        print("Accuracy on test set: " + str(correct / total))
