import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tflearn

# 原名SkFlow 改 TFLearn；集成到tf.contrib.learn

train_data = pd.read_csv('data/train.csv')
X = train_data[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare',
                'Child', 'EmbarkedF', 'DeckF', 'TitleF', 'Honor']].as_matrix()
Y = train_data[['Deceased', 'Survived']].as_matrix()

tf.app.flags.DEFINE_integer('epochs', 10, 'Training epochs')

# 创建目录
ckpt_dir = './ckpt_dir'
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

# 定义分类模型
n_features = X.shape[1]
input = tflearn.input_data([None, n_features])
y_pred = tflearn.layers.fully_connected(network, 2, activation='softmax')
net = tflearn.regression(y_pred)
model = tflearn.DNN(net)

# 读取模型存档
if os.path.isfile(os.path.join(ckpt_dir, 'model.ckpt')):
    model.load(os.path.join(ckpt_dir, 'model.ckpt'))
# 训练
model.fit(X, Y, validation_set=0.1, n_epoch=tf.app.flags.FLAGS)
# 存储模型参数
model.save(os.path.join(ckpt_dir, 'model.ckpt'))
# 查看模型在训练集上的准确性
metric = model.evaluate(X, Y)
print('Accuracy on train set: %.9f' % metric[0])


# 读取测试数据，并进行预测
test_data = pd.read_csv('test.csv')
X = test_data[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare',
               'Child', 'EmbarkedF', 'DeckF', 'TitleF', 'Honor']].as_matrix()
predictions = np.argmax(model.predict(X), 1)

submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': predictions
})
submission.to_csv('data/Titanic-sub.csv', index=False)
