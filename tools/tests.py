from __future__ import print_function
import numpy as np
import tensorflow as tf
from tools import dataset_utils

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

base_classes = ["/mnt/Storage/datasets/roshambo/tf_records/class_0",
                "/mnt/Storage/datasets/roshambo/tf_records/class_1"]
# base_classes = ["/mnt/Storage/datasets/roshambo/tf_records/shuffled_test.tfrecords"]
base_files = dataset_utils.sweep_directories(base_classes)
dataset_train = dataset_utils.dataset_tf_records(base_files, 128, "train")
iter_train, im_batch_train, lab_batch_train, lab_one_hot_train = dataset_utils.make_iterator(
    dataset_train, 2)

sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

for epoch in range(60):
    sess.run(iter_train.initializer)
    while True:
        try:
            images, labels, labels_one_hot = sess.run(
                [im_batch_train, lab_batch_train, lab_one_hot_train])
            assert not np.any(np.isnan())
        except tf.errors.OutOfRangeError:
            break

