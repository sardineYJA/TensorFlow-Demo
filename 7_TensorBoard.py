import tensorflow as tf

with tf.name_scope('input'):
    X = tf.placeholder(tf.float32, shape=[None, 6])
    y_true = tf.placeholder(tf.float32, shape=[None, 2])


with tf.name_scope('classifier'):
    weights = tf.Variable(tf.random_normal([6, 2]))
    bias = tf.Variable(tf.zeros([2]))
    y_pred = tf.nn.softmax(tf.matmul(X, weights) + bias)

    tf.summary.histogram('weights', weights)
    tf.summary.histogram('bias', bias)


with tf.name_scope('cost'):
    cross_entropy = -tf.reduce_sum(
        y_true * tf.log(y_pred + 0.000001), reduction_indices=1)
    cost = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('loss', cost)


train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost)


with tf.name_scope('accuracy'):
    correct_pred = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_pred, 1))
    acc_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    tf.summary.scalar('accuracy', acc_op)


with tf.Session() as sess:
    writer = tf.summary.FileWriter(
        './logs', sess.graph)  # 关键，之后就会生成tensorboard
    # 命令tensorboard --logdir=./logs 启动
    merged = tf.summary.merge_all()

    for step in range(max_step):
        for i in range(batch):
            # 训练过程...

            summary, accuracy = sess.run(
                [merged, acc_op], feed_dict={X: X_val, y_true: y_val})
            writer.add_summary(summary, step)
