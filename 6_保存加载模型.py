
import tensorflow as tf 

# 一次性存储
def test():
	v1 = tf.Variable(tf.zeros([200]))

	saver = tf.train.Saver() # 在Saver之后声明的变量不会被Saver处理

	v2 = tf.Variable(tf.ones([100])) # v2 不会保存

	# 训练
	with tf.Session() as sess1:
		pass

	save_path = saver.save(sess1, 'model.ckpt')

	# 加载
	with tf.Session() as sess2:
		saver.restore(sess2, 'model.ckpt') # 加载变量
		# 判别预测，或者继续训练


# 迭代存储
def test():
	v1 = tf.Variable(tf.zeros([200]))

	saver = tf.train.Saver() # 在Saver之后声明的变量不会被Saver处理

	v2 = tf.Variable(tf.ones([100])) # v2 不会保存

	# 训练
	with tf.Session() as sess:
		for step in range(max_step):
			pass

	saver.save(sess, 'my-model.ckpt', global_step=step)
	# 生成 my-model.ckpt-??? 的 checkpoint 文件
	# 其他参数：max_to_keep=5 保留最后5个 checkpoint
	# keep_checkpoint_every_n_hours=2 每2小时保存一个 checkpoint
