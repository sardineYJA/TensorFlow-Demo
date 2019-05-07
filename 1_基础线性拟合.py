import tensorflow as tf 
import numpy as np 

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0)) # 一维，[-1,1]
biases = tf.Variable(tf.zeros([1]))

y = Weights * x_data + biases

loss = tf.reduce_mean(tf.square(y-y_data))

# 建立优化器，减少误差，每次迭代都会优化
optimizer = tf.train.GradientDescentOptimizer(0.5) # 学习率<1
train = optimizer.minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	# 开始训练
	for step in range(201):
		sess.run(train)
		if step%20 == 0:
			print(step, sess.run(Weights), sess.run(biases))


