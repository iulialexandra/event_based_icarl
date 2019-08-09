import datetime
import numpy as np
import tensorflow as tf
import logging
import imageio
import os
import csv
import numpy.random as rng
from random import shuffle
from skimage.transform import resize
from tools.image_utils import ImageTransformer
from tools.image_utils import load_img, img_to_array
from itertools import chain
from tools import caffe_pb2
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from tools.misc_utils import str_to_class


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def avi_to_frame_list(avi_filename, video_limit=-1, resize_scale=None):
    """Creates a list of frames starting from an AVI movie.

    Parameters
    ----------

    avi_filename: name of the AVI movie
    gray: if True, the resulting images are treated as grey images with only
          one channel. If False, the images have three channels.
    """
    logging.info('Loading {}'.format(avi_filename))
    try:
        vid = imageio.get_reader(avi_filename, 'ffmpeg')
    except IOError:
        logging.error("Could not load meta information for file %".format(avi_filename))
        return None
    data = [im for im in vid.iter_data() if np.sum(im) > 2000]
    if data is None:
        return
    else:
        shuffle(data)
        video_limit = min(len(data), video_limit)
        assert video_limit != 0, "The video limit is 0"
        data = data[:video_limit]
        expanded_data = [np.expand_dims(im[:, :, 0], 2) for im in data]
        if resize_scale is not None:
            expanded_data = [resize(im, resize_scale, preserve_range=True) for im in expanded_data]
        logging.info('Loaded frames from {}.'.format(avi_filename))
        return expanded_data


def create_frame(y_pos, x_pos, height=240, width=304):
    """Create a single frame.

    # Arguments
    y_pos : np.ndarray
        y positions
    x_pos : np.ndarray
        x positions
    """
    histrange = [(0, v) for v in (height, width)]
    # create the frame
    img, _, _ = np.histogram2d(y_pos, x_pos, bins=(height, width), range=histrange, normed=False)

    # thresholding the events
    non_zero_img = img[np.nonzero(img)]
    mean_activation = np.mean(non_zero_img)
    std_activation = np.std(non_zero_img)
    sigma = 3 * std_activation if std_activation != 0 else 1
    # clip the image
    new_img = np.clip(img / sigma, 0, 1) * 255
    return np.expand_dims(new_img.astype(int), 2)


def load_events_bin(filename):
    """Load one binary N-Caltech recording.

    # Arguments
    filename : str
        the file name of the event binary file.
    """
    if not os.path.isfile(filename):
        raise ValueError("File {} does not exist".format(filename))

    # open data
    f = open(filename, "rb")
    raw_data = np.fromfile(f, dtype=np.uint8)

    f.close()
    raw_data = np.uint32(raw_data)

    # read  events
    all_y = raw_data[1::5]
    all_x = raw_data[0::5]
    all_p = (raw_data[2::5] & 128) >> 7  # bit 7
    all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | \
             (raw_data[4::5])

    #Process time stamp overflow events
    time_increment = 2 ** 13
    overflow_indices = np.where(all_y == 240)[0]
    for overflow_index in overflow_indices:
        all_ts[overflow_index:] += time_increment

    #Everything else is a proper td spike
    td_indices = np.where(all_y != 240)[0]
    x = all_x[td_indices]
    width = x.max() + 1
    y = all_y[td_indices]
    height = y.max() + 1
    ts = all_ts[td_indices]
    p = all_p[td_indices]
    return y, x, p, ts, width, height


def bin_to_frame_list(bin_filename, resize_scale=None):
    """Creates a list of frames starting from an bin events recording.
    """
    logging.info('Loading {}'.format(bin_filename))
    y, x, p, t, width, height = load_events_bin(bin_filename)

    s2 = np.argmax(t > 100000)
    s3 = np.argmax(t > 200000)

    f1 = np.arange(0, s2, 15000)
    f2 = np.arange(s2, s3, 15000)
    f3 = np.arange(s3, len(t), 15000)

    def process_saccade(frames):
        new_img = []
        for idx, f in enumerate(frames[:-1]):
            frame_y = y[frames[idx]:frames[idx+1]]
            frame_x = x[frames[idx]:frames[idx+1]]
            new_img.append(create_frame(frame_y, frame_x))
        return new_img

    all_images = []
    all_images.extend(process_saccade(f1))
    all_images.extend(process_saccade(f2))
    all_images.extend(process_saccade(f3))

    if resize_scale is not None:
        resized_data = [resize(im, resize_scale, preserve_range=True, anti_aliasing=True)
                        for im in all_images]
        return resized_data
    else:
        return all_images


def images_to_tfrecord(record_path, data_array, labels, class_names_dict, mode, augment,
                       augmentation_limit, **kwargs):
    lengths_file = open(os.path.join(record_path, '{}.csv'.format(mode)), 'w')
    with lengths_file:
        fields = ['class_idx', 'class_name', 'num_samples']
        csv_writer = csv.DictWriter(lengths_file, fieldnames=fields)
        csv_writer.writeheader()
        assert type(labels) == np.ndarray
        assert np.array_equal(np.array(sorted(list(class_names_dict.keys()))), np.unique(labels))
        for cls_idx in sorted(list(class_names_dict.keys())):
            filename = os.path.join(record_path, "class_{}_{}.tfrecords".format(cls_idx, mode))
            curr_label_id = np.where(labels == cls_idx)[0]
            images = data_array[curr_label_id]
            if type(images[0]) not in [str, np.str_] and augment is True:
                images = augment_dataset(images, augmentation_limit, **kwargs)
            csv_writer.writerow({'class_idx': cls_idx, 'class_name': class_names_dict[cls_idx],
                                 'num_samples': len(images)})
            tf_writer = tf.python_io.TFRecordWriter(filename)
            for image in images:
                if type(image) in [str, np.str_]:
                    image = img_to_array(load_img(image))
                    image = image / 255.
                image_dims = np.shape(image)
                image_raw = image
                image_raw = image_raw.astype(np.float32)
                image_raw = image_raw.tostring()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'height': _int64_feature(image_dims[-3]),
                    'width': _int64_feature(image_dims[-2]),
                    'depth': _int64_feature(image_dims[-1]),
                    'label': _int64_feature(int(cls_idx)),
                    'image_raw': _bytes_feature(image_raw)}))
                tf_writer.write(example.SerializeToString())
            tf_writer.close()


def save_npy_to_tfrecord(info_df, cls_idx, data_path, results_dir):
    def write_tfrecs_to_file(data, label, filename):
        image_dims = np.shape(data)
        tf_writer = tf.python_io.TFRecordWriter(filename)
        for image in data:
            image_raw = image
            image_raw = image_raw.astype(np.uint8)
            image_raw = image_raw.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(image_dims[1]),
                'width': _int64_feature(image_dims[2]),
                'depth': _int64_feature(image_dims[3]),
                'label': _int64_feature(int(label)),
                'image_raw': _bytes_feature(image_raw)}))
            tf_writer.write(example.SerializeToString())
        tf_writer.close()

    class_names = []
    train_tfrecs = []
    test_tfrecs = []
    for i, (s, symbol_info) in zip(cls_idx, info_df.iterrows()):
        label = i
        print("Converting {} to tfRecord".format(symbol_info["symbol"]))

        train_data = np.load(os.path.join(data_path, symbol_info["train_file"]))
        test_data = np.load(os.path.join(data_path, symbol_info["test_file"]))

        train_filename = os.path.join(results_dir, "class_{}_{}.tfrecords".format(label, "train"))
        write_tfrecs_to_file(train_data, label, train_filename)

        test_filename = os.path.join(results_dir, "class_{}_{}.tfrecords".format(label, "test"))
        write_tfrecs_to_file(test_data, label, test_filename)
        class_names.append(symbol_info["symbol"])
        train_tfrecs.append(train_filename)
        test_tfrecs.append(test_filename)
    return train_tfrecs, test_tfrecs, class_names


def augment_dataset(data, limit_num, **kwargs):
    transformer = ImageTransformer(**kwargs)
    num_images = len(data)

    if limit_num == 0:
        augmented_data = [transformer.random_transform(im) for im in data]
        new_data = np.concatenate((data, np.asarray(augmented_data)), axis=0)
        np.random.shuffle(new_data)
        return data[:num_images]

    elif num_images >= limit_num:
        np.random.shuffle(data)
        data = data[:limit_num]
        augmented_data = [transformer.random_transform(im) for im in data]
        new_data = np.concatenate((data, np.asarray(augmented_data)), axis=0)
        np.random.shuffle(new_data)
        return data[:num_images]

    elif num_images < limit_num:
        gap = limit_num - num_images
        image_index = rng.choice(range(num_images), size=(gap,), replace=True)
        gap_data = [transformer.random_transform(data[idx]) for idx in image_index]
        new_data = np.concatenate((data, np.asarray(gap_data)), axis=0)
        np.random.shuffle(new_data)
        return new_data


def array_to_dataset(data_array, labels_array, batch_size):
    """Creates a tensorflow dataset starting from numpy arrays.
    NOTE: Not in use.
    """
    random_array = np.arange(len(labels_array))
    rng.shuffle(random_array)
    labels = labels_array[random_array]
    data_array = tf.cast(data_array[random_array], tf.float32)
    labels = tf.cast(labels, tf.int8)
    dataset = tf.data.Dataset.from_tensor_slices((data_array, labels))
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    return dataset


def sample_class_images(data, labels):
    unique_labels = np.unique(labels)
    sampled_data = []
    for label in unique_labels:
        indices = np.where(labels == label)[0]
        im = data[indices[0]]
        sampled_data.append(im)
    return sampled_data, unique_labels


def csv_reader(filename):
    class_indices = []
    class_names = []
    num_samples = []
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_indices.append(int(row['class_idx']))
            class_names.append(row['class_name'])
            num_samples.append(int(row['num_samples']))
    return np.array(class_indices), np.array(class_names), np.array(num_samples)


def read_dataset_csv(dataset_path, val_ways):
    def join_paths(class_index, phase_data):
        return os.path.join(dataset_path, "class_{}_{}.tfrecords".format(class_index, phase_data))

    train_indices, train_names, train_num_samples = csv_reader(
        os.path.join(dataset_path,
                     "train.csv"))
    val_indices, val_names, val_samples = csv_reader(os.path.join(dataset_path,
                                                                  "test.csv"))

    # train data
    np.random.shuffle(train_indices)
    train_class_indices = train_indices[:-val_ways]
    train_class_names = train_names[train_class_indices]
    train_filenames = [join_paths(i, "train") for i in train_class_indices]

    if dataset_path.split("/")[-1] == "mini-imagenet":
        # validation data
        val_class_indices = np.random.choice(train_class_indices, val_ways, replace=False)
        val_class_names = train_names[val_class_indices]
        val_num_samples = train_num_samples[val_class_indices]
        val_filenames = [join_paths(i, "train") for i in val_class_indices]
        num_val_samples = sum(val_num_samples)

        # test data
        test_class_indices = np.random.choice(val_indices, val_ways, replace=False)
        test_class_names = val_names[test_class_indices]
        test_filenames = [join_paths(i, "test") for i in test_class_indices]
        test_num_samples = val_samples[test_class_indices]
        num_test_samples = sum(test_num_samples)
    else:
        # validation data
        val_class_indices = np.random.choice(train_class_indices, val_ways, replace=False)
        val_class_names = val_names[val_class_indices]
        val_num_samples = val_samples[val_class_indices]
        val_filenames = [join_paths(i, "test") for i in val_class_indices]
        num_val_samples = sum(val_num_samples)

        # test data
        test_class_indices = train_indices[-val_ways:]
        test_class_names = train_names[test_class_indices]
        test_filenames = [join_paths(i, "train") for i in test_class_indices]
        test_num_samples = train_num_samples[test_class_indices]
        num_test_samples = sum(test_num_samples)

    return (train_class_names, val_class_names, test_class_names, train_filenames, val_filenames,
            test_filenames, train_class_indices, val_class_indices, test_class_indices,
            num_val_samples, num_test_samples)


def parser(record):
    """It parses one tfrecord entry

    Args:
        record: image + label
    """
    features = tf.parse_single_example(record,
                                       features={
                                           'height': tf.FixedLenFeature([], tf.int64),
                                           'width': tf.FixedLenFeature([], tf.int64),
                                           'depth': tf.FixedLenFeature([], tf.int64),
                                           'image_raw': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.int64),
                                       })

    label = tf.cast(features["label"], tf.uint8)
    image_shape = tf.stack([64, 64, 1])

    # Perform additional preprocessing on the parsed data.
    image = tf.decode_raw(features["image_raw"], tf.uint8)
    image = tf.cast(image, tf.float32)
    image = tf.scalar_mul(1 / (2 ** 8), image)
    image = tf.reshape(image, image_shape)
    return image, label


def dataset_tf_records(filenames, batch_size, num_batches, mode):
    """ Starting from tfRecord files, it creates tensorflow datasets.

    Args:
        filenames: list of filenames
        batch_size: int
        num_batches: int
        mode: str; whether to build a dataset for training, incremental training or testing

    Returns:
        "inc_train" mode: the unshuffled incremental training dataset and the complete shuffled
                        dataset(incremental + exemplars)
        "train" mode: shuffled dataset
        "test" mode: unshuffled dataset
    """
    if mode == "inc_train":
        # create a dataset containing only the incremental training files
        f_inc = filenames[0]
        dataset_inc = tf.data.Dataset.from_tensor_slices(f_inc).interleave(
            lambda x: tf.data.TFRecordDataset(x).map(parser, num_parallel_calls=4),
            cycle_length=len(f_inc),
            block_length=max(1, batch_size // len(f_inc)))
        if num_batches[0] >= 0:
            real_dataset_size = batch_size * num_batches[0]
            dataset_inc = dataset_inc.take(real_dataset_size)
        dataset_inc = dataset_inc.batch(batch_size)

        # create a dataset containing both the incremental training files and the exemplars
        # from previous classes
        all_filenames = list(chain(*filenames))
        shuffle(all_filenames)
        dataset_full = tf.data.Dataset.from_tensor_slices(all_filenames).interleave(
            lambda x: tf.data.TFRecordDataset(x).map(parser, num_parallel_calls=4),
            cycle_length=len(all_filenames),
            block_length=max(1, batch_size // len(all_filenames)))
        if num_batches[0] >= 0:
            real_dataset_size = batch_size * num_batches[1]
            dataset_full = dataset_full.take(real_dataset_size)
        dataset_full = dataset_full.shuffle(batch_size * 5)
        dataset_full = dataset_full.batch(batch_size)
        dataset_full = dataset_full.prefetch(batch_size * 10)
        return dataset_inc, dataset_full

    elif mode == "train":
        shuffle(filenames)
        dataset = tf.data.Dataset.from_tensor_slices(filenames
                                                     ).interleave(
            lambda x: tf.data.TFRecordDataset(x).map(parser, num_parallel_calls=4),
            cycle_length=len(filenames),
            block_length=max(1, batch_size // len(filenames)))
        if num_batches >= 0:
            real_dataset_size = batch_size * num_batches
            dataset = dataset.take(real_dataset_size)
        dataset = dataset.shuffle(buffer_size=batch_size * 5)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(batch_size * 10)
        return dataset

    elif mode == "test":
        dataset = tf.data.Dataset.from_tensor_slices(filenames
                                                     ).interleave(
            lambda x: tf.data.TFRecordDataset(x).map(parser, num_parallel_calls=4),
            cycle_length=len(filenames),
            block_length=max(1, batch_size // len(filenames)))
        if num_batches >= 0:
            real_dataset_size = batch_size * num_batches
            dataset = dataset.take(real_dataset_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(batch_size * 10)
        return dataset

    else:
        "Please specify a different mode"


def dataset_array(data_array, labels_array, batch_size, shuffle_size):
    """Creates a tensorflow dataset starting from numpy arrays.
    NOTE: Not in use.
    """
    labels = tf.cast(labels_array, tf.int32)
    dataset = tf.data.Dataset.from_tensor_slices((data_array, labels))
    dataset = dataset.shuffle(buffer_size=shuffle_size)
    dataset = dataset.batch(batch_size)
    return dataset


def make_iterator(dataset, num_classes):
    """Makes a tensorflow iterator for the given dataset.

    Args:
        dataset: tensorflow dataset object
        num_classes: int; how many different classes to use for one-hot encoding of the labels
    """
    iterator = dataset.make_initializable_iterator()
    image_batch, label_batch = iterator.get_next()
    labels_one_hot = tf.one_hot(label_batch, num_classes)
    return iterator, image_batch, label_batch, labels_one_hot


def prepare_networks_train(model, image_batch, nb_classes, res_blocks):
    """Builds networks for training and distillation.

    Args:
        model: str; type of network model to use
        image_batch: tf tensor; bunch of images to be processed at once
        nb_classes: int; number of output nodes for the network
        res_blocks: int; if using ResNets, how many residuals blocks to build into the network

    Returns:
        weights for the training and distillation networks + outputs for both networks
    """
    network = str_to_class(model)
    scores = []
    with tf.variable_scope('Net') as scope:
        net = network(nb_classes, image_batch, "train", res_blocks)
        score, _ = net.build_cnn()
        scores.append(score)
        scope.reuse_variables()
    variables_graph = tf.get_collection(tf.GraphKeys.WEIGHTS, scope='Net')

    scores_stored = []
    with tf.variable_scope('stored_Net') as scope:
        net = network(nb_classes, image_batch, "test", res_blocks)
        score, _ = net.build_cnn()
        scores_stored.append(score)
        scope.reuse_variables()
    variables_graph_stored = tf.get_collection(tf.GraphKeys.WEIGHTS, scope='stored_Net')
    return variables_graph, variables_graph_stored, scores, scores_stored


def prepare_network_test(model, image_batch, nb_classes, params_to_load, res_blocks):
    """Builds networks for testing.

    Args:
        model: str; type of network model to use
        image_batch: tf tensor; bunch of images to be processed at once
        nb_classes: int; number of output nodes for the network
        params_to_load: trained network weights to use for inference
        res_blocks: int; if using ResNets, how many residuals blocks to build into the network

    Returns:
        weight initialization op, the outputs of the network, the tf op for retrieving the outputs
        of the feature map before the output layer
    """
    network = str_to_class(model)
    with tf.variable_scope('test_Net') as scope:
        net = network(nb_classes, image_batch, "test", res_blocks)
        scores, _ = net.build_cnn()
        operations = tf.get_default_graph().get_operations()
        pool_last_ops = [operation for operation in operations if
                         "pool_last" in operation.name and "test_Net" in operation.name]
        assert len(pool_last_ops) == 1, "You should only have one op in the last pooling layer"
        op_feature_map = pool_last_ops[0].outputs[0]
        scope.reuse_variables()
    variables_test = tf.get_collection(tf.GraphKeys.WEIGHTS, scope='test_Net')
    inits = net.get_weight_initializer(params_to_load, variables_test)
    return inits, scores, op_feature_map


def save_for_inference(weights_folder, weights, model, nb_classes, class_names, res_blocks, level):
    """Saves the network ready for inference

    Args:
        weights_folder: path to network model
        weights: the weights to save
        model: type of network to use
        nb_classes: number of network outputs
        res_blocks: how many residual blocks to use when building the network
        level: whether we want to save the entire network or only the features before the output

    Returns:
        the file path to the saved network
    """
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    X = tf.placeholder(dtype=tf.float32, shape=(
        None, 64, 64, 1), name='input')
    network = str_to_class(model)
    if level == "features":
        net = network(nb_classes, X, "icarl_inference", res_blocks)
        feature_maps = net.build_cnn()
        feature_maps = tf.identity(feature_maps, name="output")
        variables_graph = tf.get_collection(tf.GraphKeys.WEIGHTS)
        op_assign = [(variables_graph[i]).assign(weights[i]) for i in
                     range(len(weights) - 2)]
    elif level == "logits":
        net = network(nb_classes, X, "test", res_blocks)
        out, feature_maps = net.build_cnn()
        out = tf.identity(out, name="output")
        variables_graph = tf.get_collection(tf.GraphKeys.WEIGHTS)
        op_assign = [(variables_graph[i]).assign(weights[i]) for i in
                     range(len(weights))]
    sess.run(op_assign)
    graph_def = sess.graph.as_graph_def()
    # for node in graph_def.node:
    #     if node.op == 'RefSwitch':
    #         node.op = 'Switch'
    #         for index in range(len(node.input)):
    #             if 'moving_' in node.input[index]:
    #                 node.input[index] = node.input[index] + '/read'
    #     elif node.op == 'AssignSub':
    #         node.op = 'Sub'
    #         if 'use_locking' in node.attr: del node.attr['use_locking']
    #     elif node.op == 'AssignAdd':
    #         node.op = 'Add'
    #         if 'use_locking' in node.attr: del node.attr['use_locking']
    constant_graph = graph_util.convert_variables_to_constants(sess, graph_def,
                                                               ["output"])
    graph_io.write_graph(constant_graph, weights_folder, "model.pb", as_text=False)
    labels = list(chain(*class_names))
    np.savetxt(os.path.join(weights_folder, "labels.txt"), X=labels, delimiter="\n",
               fmt="%s")


def load_in_feature_space(im_batch, lab_batch, lab_one_hot, op_feature_map, sess):
    """Runs images through network to compute the feature maps of the last layer before output

    Args:
        im_batch: tf tensor; bunch of images to be processed at once
        lab_batch: tf tensor; labels of images to be processed at once
        lab_one_hot: tf tensor, one-hot encoded
        op_feature_map: tf op for retrieving the outputs of the feature map before the output layer
        sess:tensorflow session

    Returns: image batch, labels and one-hot encoded labels batch, feature maps

    """
    images_batch, labels_batch, labels_one_hot_batch, feat_map_tmp = sess.run([im_batch,
                                                                               lab_batch,
                                                                               lab_one_hot,
                                                                               op_feature_map])
    mapped_prototypes = feat_map_tmp[:, 0, 0, :]
    return images_batch, labels_batch, labels_one_hot_batch, (
            mapped_prototypes.T / np.linalg.norm(mapped_prototypes.T, axis=0))


def write_lmdb_tfrecords(open_lmdb, save_dir, num_classes):
    """Convert lmdb data to tfRecords"""
    with open_lmdb.begin() as txn:
        cursor = txn.cursor()
        now = datetime.datetime.now()
        date = "{}_{}_{}-{}_{}_{}_{}".format(now.year, now.month, now.day, now.hour, now.minute,
                                             now.second, now.microsecond)
        class_dirs = []
        for i in range(num_classes):
            class_dirs.append(os.path.join(save_dir, "class_{}".format(i)))
            if not os.path.exists(class_dirs[i]):
                os.makedirs(class_dirs[i])

        filenames = [os.path.join(class_dirs[i], "{}.tfrecords".format(date)) for i in
                     range(num_classes)]

        writers = [tf.python_io.TFRecordWriter(filenames[i]) for i in range(num_classes)]
        for key, value in cursor:
            datum = caffe_pb2.Datum()
            datum.ParseFromString(value)
            flat_x = np.fromstring(datum.data, dtype=np.uint8)
            x = flat_x.reshape(datum.height, datum.width, datum.channels)
            y = datum.label
            image_raw = x.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(datum.height),
                'width': _int64_feature(datum.width),
                'depth': _int64_feature(datum.channels),
                'label': _int64_feature(int(y)),
                'image_raw': _bytes_feature(image_raw)}))
            writers[y].write(example.SerializeToString())
    for writer in writers:
        writer.close()


def exemplars_to_tfrecord(data_list, labels_list, classes, save_path):
    """Dump computed exemplars to a tfRecord file

    Args:
        data_list: float list; exemplars data
        labels_list: int list; exemplars labels
        classes: int list; which classes we are using
        save_path: where to save the exemplars tfRecords

    Returns:
        the filenames of the saved exemplars
    """
    path = os.path.join(save_path, "exemplars_" + str(classes))
    if not os.path.exists(path):
        os.makedirs(path)
    filenames = []
    for idx, class_exemplars in enumerate(data_list):
        filename = os.path.join(path, "exemplars_class_{}.tfrecords".format(classes[idx]))
        filenames.append(filename)
        writer = tf.python_io.TFRecordWriter(filename)
        for image, label in zip(class_exemplars, labels_list[idx]):
            image_raw = image * 256
            image_raw = image_raw.astype(np.uint8)
            image_raw = image_raw.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(64),
                'width': _int64_feature(64),
                'depth': _int64_feature(1),
                'label': _int64_feature(int(label)),
                'image_raw': _bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())
        writer.close()
    return filenames
