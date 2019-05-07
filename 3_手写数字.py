import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 下载到文件夹mnist_data中，存在则直接打开
mnist = input_data.read_data_sets('mnist_data', one_hot=True)


print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)

"""
(55000, 784) (55000, 10)
(10000, 784) (10000, 10)
(5000, 784) (5000, 10)
"""

# 添加层


def add_layer(inputs, in_size, out_size, activation_function=None):
    # Weights 是一个矩阵，[行，列]为[in_size, out_size]
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))  # 正态分布

    # 初始值推荐不为0
    biases = tf.Variable(tf.zeros([10]) + 0.1)

    # Weights *x + b的初始值
    Wx_plus_b = tf.matmul(inputs, Weights) + biases

    # 激活
    if activation_function is None:
        # 没有激活函数就是线性函数
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)

    return outputs


def compute_accuracy(v_xs, v_ys):
    # 全局变量
    global prediction
    # 生成预测值，也是概率，即每个数字的概率
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})

    # 对比预测的数据是否和真实值相等，对比位置是否相等，相等就对了
    # tf.arg_max 是从一个tensor中寻找最大值的序号
    # tf.arg_max(y_pre, 1)求概率最大的预测数字
    # tf.arg_max(v_ys, 1)找样本的真实数字类别
    correct_prediction = tf.equal(tf.arg_max(y_pre, 1), tf.arg_max(v_ys, 1))

    # 计算多少个对，多少个错
    # tf.cast(x, dtype)   # 将 x 数据转换为dtype类型
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})

    return result


# 不规定有多少个sample，但是每个sample大小为784（28*28）
xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])

prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)

# 熵 = -累加符号y*log(y)
cross_entropy = tf.reduce_mean(
    -tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
train_strp = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(2000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_strp, feed_dict={xs: batch_xs, ys: batch_ys})

        if i % 50 == 0:
            print(compute_accuracy(mnist.test.images, mnist.test.labels))
