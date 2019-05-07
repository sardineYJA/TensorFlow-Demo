import random
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import core_rnn
from tensorflow.contrib.rnn.python.ops import core_rnn_cell


def build_data(n):
    xs = []   # 存储多个列表，每个列表表示一个向量，每个向量为n维
    ys = []   # 存储多个列表，每个列表只有一个数值，表示对应xs的标签

    for i in range(2000):
        k = random.uniform(1, 50)  # 返回一个[1-50]大小的浮点数
        x = [[np.sin(k + j)] for j in range(0, n)]
        y = [np.sin(k + n)]

        xs.append(x)
        ys.append(y)

    train_x = np.array(xs[0:1500])
    train_y = np.array(ys[0:1500])

    test_x = np.array(xs[1500:])
    test_y = np.array(ys[1500:])

    return train_x, train_y, test_x, test_y


def test():
    length = 10
    time_step_size = length
    vector_size = 1
    batch_size = 10
    test_size = 10

    X = tf.placeholder('float', [None, length, vector_size])
    Y = tf.placeholder('float', [None, 1])
    # stddev正态分布的标准差
    W = tf.Variable(tf.random_normal([10, 1], stddev=0.01))
    B = tf.Variable(tf.random_normal([1], stddev=0.01))

    def seq_predict_model(X, w, b, time_step_size, vector_size):
        X = tf.transpose(X, [1, 0, 2]) # 交换输入张量的不同维度，1和2维度交换
        X = tf.reshape(X, [-1, vector_size])
        X = tf.split(X, time_step_size, 0)

        cell = core_rnn_cell.BasicRNNCell(num_units=10)
        initial_state = tf.zeros([batch_size, cell.state_size])
        outputs, _states = core_rnn.static_rnn(
            cell, X, initial_state=initial_state)

        # 线性激活函数
        return tf.matmul(outputs[-1], w) + b, cell.state_size

    pred_y, _ = seq_predict_model(X, W, B, time_step_size, vector_size)

    # 代价函数，优化算法
    loss = tf.square(tf.subtract(Y, pred_y))
    train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

    # 构造正弦序列进行学习
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(50):
            for end in range(batch_size, len(train_x), batch_size):
                begin = end - batch_size
                x_value = train_x[begin:end]
                y_value = train_y[begin:end]
                sess.run(train_op, feed_dict={X: x_value, Y: y_value})

            test_indices = np.arange(len(test_x))
            np.random.shuffle(test_indices)          # 打乱顺序
            test_indices = test_indices[0:test_size]
            x_value = test_x[test_indices]
            y_value = test_y[test_indices]

            val_loss = np.mean(
                sess.run(loss, feed_dict={X: x_value, Y: y_value}))
            print('Run %s' % i, val_loss)


if __name__ == '__main__':
    build_data(10)  # 10表示向量长度

    print('OK')
