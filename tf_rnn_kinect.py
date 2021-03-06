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

training_loader = KinectDataLoader()
test_loader = KinectDataLoader('./Kinect_test/')

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

def test_accuracy(test_set=False):
    total = 0
    correct = 0
    target = None
    if test_set:
        target = test_loader
    else:
        target = training_loader
    for test_xs, test_ys in target.all_batches(batch_size):
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
    return correct / total

def RNN(X, weights, biases):
    # 为了方便计算隐层输入
    X = tf.reshape(X, [-1, n_inputs])  # 将 shape=(128批, 200帧, 63) 的动作序列转换为 shape=(128*200, 63)
    
    # linear activated hidden layer
    X_in = tf.matmul(X, weights['in']) + biases['in']  # 计算出隐层输入
    X_in = tf.reshape(X_in, [batch_size, -1, n_hidden_units])  # 将隐层输入转换为 shape=(128批, 200帧, 128隐层神经元数)
    # X_in = tf.nn.dropout(X_in, keep_prob=0.9)

    # lstm单元
    lstm_cell = tf.nn.rnn_cell.LSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state, time_major=False)

    results = tf.matmul(states[1], weights['out']) + biases['out']  # shape=(128批, 5种动作)
    return results

pred = RNN(x, weights, biases)  # 预测
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))  # 代价
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)) + tf.reduce_sum(tf.pow(weights['out'], 2))  # 代价(正则化)
train_op = tf.train.AdamOptimizer(lr).minimize(cost)  # 训练方法

init = tf.initialize_all_variables()
saver = tf.train.Saver(max_to_keep=1)

# fig = plt.figure(figsize=(8, 10))
# ax = fig.add_subplot(2, 1, 1)
# ay = fig.add_subplot(2, 1, 2)
# ax.set_xlabel('steps')
# ax.set_ylabel('cost')
# ay.set_xlabel('steps')
# ay.set_ylabel('accuracy on training set')
# plt.ion()
# plt.show()

# graph_steps = []
# graph_cost = []
# train_set_acc = []
# test_set_acc = []
# ax.plot(graph_steps, graph_cost, color='r', label='cost')
# ay.plot(graph_steps, train_set_acc, color='g', label='ac on train')
# ay.plot(graph_steps, test_set_acc, color='b', label='ac on test')
# plt.legend()

with tf.Session() as sess:

    if choice == 1:
        sess.run(init)
        # model_file = tf.train.latest_checkpoint('ckpt_rnn/')
        # saver.restore(sess, model_file)
        step = 0
        accuracy = 0
        while True:
            batch_xs, batch_ys = training_loader.next_batch(batch_size)  # 读进来的数据 shape=(128 batch_size, 200帧, 21个点, 3 xyz)
            batch_xs = batch_xs.reshape([batch_size, n_steps, 63])  # 把上述 (128, 200, 21, 3) 转为 (128, 200, 63)
            sess.run([train_op], feed_dict={
                x: batch_xs,
                y: batch_ys
            })
            if step % 10 == 0:
                # graph_steps.append(step)
                cst = sess.run(cost, feed_dict={
                    x: batch_xs,
                    y: batch_ys
                })
                # graph_cost.append(cst)
                # ax.plot(graph_steps, graph_cost, color='r', label='cost')
                train_acc = test_accuracy()
                # train_set_acc.append(train_acc)
                test_acc = test_accuracy(True)
                # test_set_acc.append(test_acc)
                print("Cost: " + str(cst))
                print("Accuracy on training set: " + str(train_acc))
                print("Accuracy on test set: " + str(test_acc))
                # ay.plot(graph_steps, train_set_acc, color='g', label='ac on train')
                # ay.plot(graph_steps, test_set_acc, color='b', label='ac on test')
                saver.save(sess, 'ckpt_rnn/trained.ckpt', global_step=step)
                # plt.pause(0.1)
            step += 1

    elif choice == 2:
        model_file = tf.train.latest_checkpoint('ckpt_rnn/')
        saver.restore(sess, model_file)
        total = 0
        correct = 0
        for test_xs, test_ys in test_loader.all_batches(batch_size):
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

"""
正则化: 训练19900次, 得到的结果同时满足精确度高和代价小
"""
