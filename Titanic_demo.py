
import pandas as pd
import tensorflow as tf
import numpy as np

data = pd.read_csv('train.csv')
print(data.info())


# 选6个特征，apply()将该列每行进行lambda函数运算
data['Sex'] = data['Sex'].apply(lambda s: 1 if s == 'male' else 0)
data = data.fillna(0)
dataset_X = data[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare']]
dataset_X = dataset_X.as_matrix()  # 返回类型ndarray

# 标签
data['Deceased'] = data['Survived'].apply(lambda s: int(not s))
dataset_Y = data[['Deceased', 'Survived']]
dataset_Y = dataset_Y.as_matrix()  # 返回类型ndarray

# 切割
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    dataset_X, dataset_Y, test_size=0.2, random_state=42)

print(X_train[:10])
print(y_train[:10])
print(X_test[:10])
print(y_test[:10])

# 逻辑回归分类器：y` = softmax(xW+b)
X = tf.placeholder(tf.float32, shape=[None, 6])  # 6个特征
y = tf.placeholder(tf.float32, shape=[None, 2])  # 2个标签
W = tf.Variable(tf.random_normal([6, 2]), name='weights')
b = tf.Variable(tf.zeros([2]), name='bias')

y_pred = tf.nn.softmax(tf.matmul(X, W) + b)


# 代价函数:cross entropy:C = -1/n  * 累加符号(y * logy`)
# log(0)则输出非法，loss全为nan，必须加极小值
cross_entropy = - \
    tf.reduce_sum(y * tf.log(y_pred + 0.000001), reduction_indices=1)
cost = tf.reduce_mean(cross_entropy)


# 优化
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)


# 训练迭代
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(10):
        total_loss = 0.
        for i in range(len(X_train)):
            feed = {X: [X_train[i]], y: [y_train[i]]}

            _, loss = sess.run([train_op, cost], feed_dict=feed)
            total_loss += loss
        print('Epoch : %04d, total loss = %.9f' % (epoch + 1, total_loss))
    print('Training complete!')

    # 测试模型
    pred = sess.run(y_pred, feed_dict={X: X_test})
    correct = np.equal(np.argmax(pred, 1), np.argmax(y_test, 1))
    accuracy = np.mean(correct.astype(np.float32))
    print('Accuracy on validation set : %.9f' % accuracy)

    print(np.argmax(pred, 1))
    print(correct)

"""
	函数介绍:
	np.argmax(array, axis=0) 返回每列最大值的下标索引
	np.argmax(array, axis=1) 返回每行最大值的下标索引

"""


# 对 test.csv 生成预测结果
def test():
    testdata = pd.read_csv('test.csv')
    testdata = testdata.fillna(0)
    testdata['Sex'] = testdata['Sex'].apply(lambda s: 1 if s == 'male' else 0)

    X_test = testdata[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare']]

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, 'model.ckpt')

    predictions = np.argmax(sess.run(y_pred, feed_dict={X: X_test}), 1)

    submission = pd.DataFrame({
        'PassengerId': testdata['PassengerID'],
        'Survived': predictions
    })
    submission.to_csv('Titanic-submission.csv', index=False)
