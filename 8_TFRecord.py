import tensorflow as tf
import pandas as pd

# 大数据处理推荐TFRecord 文件


# 将train.csv文件转换 train.tfrecords
def transform_to_tfrecord():
    data = pd.read_csv('train.csv')
    tfrecord_file = 'train.tfrecords'

    def int_feature(value):
        return tf.train.Feature(
            int64_list=tf.train.Int64List(value=[value]))

    def float_feature(value):
        return tf.train.Feature(
            float_list=tf.train.FloatList(value=[value]))

    writer = tf.python_io.TFRecordWriter(tfrecord_file)

    for i in range(len(data)):
        features = tf.train.Features(feature={
            'Age': float_feature(data['Age'][i]),
            'Survived': int_feature(data['Survived'][i]),
            'Pclass': int_feature(data['Pclass'][i]),
            'Parch': int_feature(data['Parch'][i]),
            'SibSp': int_feature(data['SibSp'][i]),
            'Sex': int_feature(1 if data['Sex'][i] == 'male' else 0),
            'Fare': float_feature(data['Fare'][i])
        })
        example = tf.train.Example(features=features)
        writer.write(example.SerializeToString())
    writer.close()
    print('转换TFRecord成功')


# 读取 tfrecord 文件
def read_and_decode(train_files, num_threads=2, num_epochs=100, batch_size=10, min_after_dequeue=10):

    reader = tf.TFRecordReader()
    # 定义TFRecord文件作为模型结构的输入部分
    filename_queue = tf.train.string_input_producer(
        train_files, num_epochs=num_epochs)

    _, Serialized_example = reader.read(filename_queue)
    featuresdict = tf.parse_single_example(
        Serialized_example,
        features={
            'Age': tf.FixedLenFeature([], tf.float32),
            'Survived': tf.FixedLenFeature([], tf.int64),
            'Pclass': tf.FixedLenFeature([], tf.int64),
            'Parch': tf.FixedLenFeature([], tf.int64),
            'SibSp': tf.FixedLenFeature([], tf.int64),
            'Sex': tf.FixedLenFeature([], tf.int64),
            'Fare': tf.FixedLenFeature([], tf.float32)})

    labels = featuresdict.pop('Survived')
    features = [tf.cast(value, tf.float32) for value in featuresdict.values()]

    features, labels = tf.train.shuffle_batch(
        [features, labels],
        batch_size=batch_size,
        num_threads=num_threads,
        capacity=min_after_dequeue + 3 * batch_size,
        min_after_dequeue=min_after_dequeue)

    return features, labels


def train_with_queuerunner():
    x, y = read_and_decode(['train.tfrecords'])

    with tf.Session() as sess:
        tf.group(tf.global_variables_initializer(),
                 tf.local_variables_initializer()).run()

        coord = tf.train.Coordinator()  # 负责实现数据输入线程的同步
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # 函数start_queue_runners开启对应运行会话Session的所有线程队列并返回线程句柄

        try:
            step = 0
            while not coord.should_stop():
                features, labels = sess.run([x, y])
                if step % 100 == 0:
                    print('step %d:' % step, labels)
                step += 1

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()

        coord.join(threads)


if __name__ == '__main__':

    # transform_to_tfrecord()

    features, labels = read_and_decode(['train.tfrecords'])
    print(features)
    print(labels)

    # train_with_queuerunner()

    print('OK')
