import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # warnning

# print('--------------------')
# with tf.device("/cpu:0"):
#     a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
#     b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
#     c = tf.matmul(a, b)
#     # 查看计算时硬件的使用情况
#     sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
#     print(sess.run(c))
#     print(c)
#     sess.close()
#     print('CPU working')

print('--------------------')
with tf.device("/gpu:0"):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
    c = tf.matmul(a, b)
    # 查看计算时硬件的使用情况
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    print(sess.run(c))
    print(c)
    sess.close()
    print('GPU working')
