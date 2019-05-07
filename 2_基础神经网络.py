import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt


# 添加层
def add_layer(inputs, in_size, out_size, activation_function=None):
    # Weights 是一个矩阵，[行，列]为[in_size, out_size]
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))  # 正态分布

    # 初始值推荐不为0
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)

    # Weights *x + b的初始值
    Wx_plus_b = tf.matmul(inputs, Weights) + biases

    # 激活
    if activation_function is None:
        # 没有激活函数就是线性函数
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)

    return outputs


# 定义数据形式

# (-1,1)之间有300个单位，后面是维度，x_data有300行
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
print(x_data)

# 加噪声，均值为0，方差为0.05，大小和x_data一样
noise = np.random.normal(0, 0.05, x_data.shape)

y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])


# 建立网络

# 定义隐藏层，输入1个节点，输出10个节点
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)

# 定义输出层
prediction = add_layer(l1, 10, 1, activation_function=None)


# 预测

# 损失函数，算出的是每个例子的平方，要求和（reduction_indices=[1],按行求和），再求均值
loss = tf.reduce_mean(tf.reduce_sum(
    tf.square(ys - prediction), reduction_indices=[1]))


# 训练

# 优化算法，minimize(loss)以0.1的学习率
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()


# with tf.Session() as sess:
#     sess.run(init)
#     for i in range(1000):
#         sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
#         if i % 50 == 0:
#             print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))


# 可视化
with tf.Session() as sess:
    sess.run(init)

    # 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x_data, y_data)
    plt.ion()      # 不暂停
    # plt.show()   # 绘制一次就会暂停

    for i in range(1000):
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % 50 == 0:
            # print(sess.run(loss, feed_dict={xs:x_data, ys:y_data}))

            # 第一次没有线，会报错，try忽略
            try:
                # 画出一条后抹除掉，去除第一个线段
                ax.lines.remove(lines[0])
                prit('-----')
            except Exception:
                pass

            prediction_value = sess.run(prediction, feed_dict={xs: x_data})
            lines = ax.plot(x_data, prediction_value, 'r_', lw=5)  # lw线宽

            plt.pause(1)  # 暂停1s
