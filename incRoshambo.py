from __future__ import print_function
import numpy as np
import tensorflow as tf
import pickle
from tools import dataset_utils
import logging
import bisect
import os
import time
from itertools import chain
from scipy.spatial.distance import cdist

logger = logging.getLogger("roshambo_demo")

np.random.seed(1)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


class IncRoshambo(object):
    """Incremental learning algorithm. It performs base training, incremental training and
     evaluation.
    """

    def __init__(self, args):
        self.image_dims = args.image_dims
        self.batch_size = args.batch_size
        self.network = args.network
        self.res_blocks = args.res_blocks
        self.base_epochs = args.base_epochs
        self.inc_epochs = args.inc_epochs
        self.num_batches_base = args.num_batches_base
        self.num_batches_inc = args.num_batches_inc
        self.num_batches_eval = args.num_batches_eval
        self.lr_old = args.lr_old
        self.lr_factor = args.lr_factor
        self.weight_decay = args.weight_decay
        self.lr_reduce = args.lr_reduce
        self.nb_protos_per_class = args.exemplars_memory
        self.nb_classes = 0
        self.all_classes = []
        self.class_names = []
        self.class_full_means = []
        self.class_exemplar_means = []

        self.exemplar_filenames = []
        self.save_path = args.save_path
        self.base_chkpt = args.base_chkpt
        self.inc_chkpt = args.inc_chkpt
        self.results_path = args.results_path
        self.current_save_dir = args.results_path

    def base_train(self, base_train_data, base_classes, base_cls_names):
        """Performs base training.

        Args:
            base_train_data: filenames of the tfRecords used for training
            base_classes: int list; indices of the classes used for base training.
                          Counting starts at 0.
            base_cls_names: str list; names of the base classes
        """
        self.current_save_dir = os.path.join(self.results_path, "base")
        weights_folder = os.path.join(self.current_save_dir, "weights")
        if not os.path.exists(weights_folder):
            os.makedirs(weights_folder)

        self.all_classes.append(base_classes)
        self.class_names.append(base_cls_names)
        self.nb_classes += len(base_classes)

        # create the input
        dataset_train = dataset_utils.dataset_tf_records(base_train_data, self.batch_size,
                                                         self.num_batches_base, "train")
        iter_train, im_batch_train, lab_batch_train, lab_one_hot_train = dataset_utils.make_iterator(
            dataset_train, self.nb_classes)
        im_batch_train = tf.identity(im_batch_train, name="input")
        variables_graph, variables_graph2, scores, scores_stored = dataset_utils.prepare_networks_train(
            self.network, im_batch_train, self.nb_classes, self.res_blocks)
        scores = tf.concat(scores, 0)
        scores = tf.identity(scores, name="output")
        l2_reg = self.weight_decay * tf.reduce_sum(
            tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='Net'))
        loss_classif = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=lab_one_hot_train, logits=scores))
        loss = loss_classif + l2_reg
        learning_rate = tf.placeholder(tf.float32, shape=[])
        opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
        train_step = opt.minimize(loss, var_list=variables_graph)
        saver = tf.train.Saver()

        # Run the learning phase
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())

        #
        # total_parameters = 0
        # for variable in tf.trainable_variables():
        #     shape = variable.get_shape()
        #     print(shape)
        #     print(len(shape))
        #     variable_parameters = 1
        #     for dim in shape:
        #         print(dim)
        #         variable_parameters *= dim.value
        #     print(variable_parameters)
        #     total_parameters += variable_parameters
        # print(total_parameters)

        lr = self.lr_old
        loss_batch = []
        if self.base_chkpt is not None:
            saver.restore(sess, self.base_chkpt)

        logger.info("********************************")
        logger.info("Base training of classes {} over {} epochs".format(base_cls_names,
                                                                        self.base_epochs))
        logger.info("********************************")

        for epoch in range(self.base_epochs):
            logger.info('Epoch {}'.format(epoch))
            sess.run(iter_train.initializer)
            batch = 0
            while True:
                try:
                    loss_train, _, sc, lab = sess.run([loss_classif, train_step, scores,
                                                       lab_batch_train],
                                                      feed_dict={learning_rate: lr})
                    loss_batch.append(loss_train)
                    if len(loss_batch) == min(self.num_batches_base, 100):
                        logger.info("Training error: {}".format(np.mean(loss_batch)))
                        loss_batch = []
                    batch += 1
                except tf.errors.OutOfRangeError:
                    break

            # Print the training accuracy every epoch
            stat = []
            stat += ([ll in best for ll, best in zip(lab, np.argsort(sc, axis=1)[:, -1:])])
            stat = np.average(stat)
            logger.info('Training accuracy at epoch {}: {}'.format(epoch, stat))

            # Decrease the learning rate
            if epoch > 0 and ((epoch + 1) % self.lr_reduce) == 0:
                lr /= self.lr_factor
                saver.save(sess, os.path.join(weights_folder, 'base_model'), global_step=epoch)

        # Extract weights
        self.save_weights = sess.run([variables_graph[i] for i in range(len(variables_graph))])

        # Save exemplars for the base training data
        self.exemplars_management(dataset_train, sess)
        sess.close()
        tf.reset_default_graph()

        # Create inference model file
        dataset_utils.save_for_inference(weights_folder, self.save_weights,
                                         self.network, self.nb_classes,
                                         self.class_names,
                                         self.res_blocks, "logits")

        class_exemplar_means_file = os.path.join(self.current_save_dir, "means.txt")
        np.savetxt(class_exemplar_means_file, X=self.class_exemplar_means,
                   delimiter=" ", newline="\n", fmt="%.18f")

        # Save the entire incremental learning object
        with open(os.path.join(self.current_save_dir, "IncStone_algo.pickle"),
                  "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        logger.info("Finished training base classes. Saved algorithm to file")

    def regular_train(self, inc_train_data, inc_classes, inc_cls_names):
        """Performs incremental training without distillation, exemplars and distance to mean.

        Args:
            inc_train_data: filenames of the tfRecords used for training
            inc_classes: int list; indices of the classes used for incremental training.
            inc_cls_names: str list; names of the incremental classes
        """
        self.all_classes.append(inc_classes)
        self.class_names.append(inc_cls_names)
        self.nb_classes += len(inc_classes)
        self.current_save_dir = os.path.join(self.results_path,
                                             "regular_inc_{}".format(len(self.all_classes) - 1))
        if not os.path.isdir(self.current_save_dir):
            os.makedirs(self.current_save_dir)

        weights_folder = os.path.join(self.current_save_dir, "weights")
        if not os.path.exists(weights_folder):
            os.makedirs(weights_folder)

        dataset_train = dataset_utils.dataset_tf_records(inc_train_data, self.batch_size,
                                                         self.num_batches_inc, "train")

        iter_train, im_batch_train, lab_batch_train, lab_one_hot_train = dataset_utils.make_iterator(
            dataset_train, self.nb_classes)
        im_batch_train = tf.identity(im_batch_train, name="input")
        variables_graph, variables_graph2, scores, scores_stored = dataset_utils.prepare_networks_train(
            self.network, im_batch_train, self.nb_classes, self.res_blocks)
        scores = tf.concat(scores, 0)
        scores = tf.identity(scores, name="output")
        l2_reg = self.weight_decay * tf.reduce_sum(
            tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='Net'))
        loss_classif = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=lab_one_hot_train, logits=scores))
        loss = loss_classif + l2_reg
        learning_rate = tf.placeholder(tf.float32, shape=[])
        opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
        train_step = opt.minimize(loss, var_list=variables_graph)
        saver = tf.train.Saver()

        # Run the learning phase
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        lr = self.lr_old
        loss_batch = []

        b_temp = tf.get_variable("b_temp",
                                 shape=np.shape(self.save_weights[-1])[:-1] + (len(inc_classes),),
                                 initializer=tf.contrib.layers.xavier_initializer())
        self.save_weights[-1] = tf.concat([self.save_weights[-1], b_temp], axis=-1)

        W_temp = tf.get_variable("W_temp",
                                 shape=np.shape(self.save_weights[-2])[:-1] + (len(inc_classes),),
                                 initializer=tf.contrib.layers.xavier_initializer())
        self.save_weights[-2] = tf.concat([self.save_weights[-2], W_temp], axis=-1)
        sess.run(b_temp.initializer)
        sess.run(W_temp.initializer)

        # Run the loading of the weights for the learning network
        sess.run([(variables_graph[i]).assign(self.save_weights[i]) for i in
                  range(len(self.save_weights))])
        if self.inc_chkpt is not None:
            saver.restore(sess, self.inc_chkpt)

        logger.info("********************************")
        logger.info("Regular incremental training on classes {} over {} epochs"
                    "".format(inc_classes, self.inc_epochs))
        logger.info("********************************")

        for epoch in range(self.inc_epochs):
            logger.info('Epoch {}'.format(epoch))
            sess.run(iter_train.initializer)
            np.save(os.path.join(self.current_save_dir, "biases_epoch_{}".format(epoch)),
                       variables_graph[-1].eval(session=sess))

            np.save(os.path.join(self.current_save_dir, "weights_epoch_{}".format(epoch)),
                       variables_graph[-2].eval(session=sess))
            batch = 0
            while True:
                try:
                    # begin = time.time()
                    loss_train, _, sc, lab = sess.run(
                        [loss_classif, train_step, scores, lab_batch_train],
                        feed_dict={learning_rate: lr})
                    loss_batch.append(loss_train)
                    if len(loss_batch) == min(self.num_batches_inc, 100):
                        logger.info("Training error: {}".format(np.mean(loss_batch)))
                        loss_batch = []
                    # end = time.time()
                    # logger.debug("Took {} second to train on one batch".format(end - begin))
                    batch += 1
                except tf.errors.OutOfRangeError:
                    break

            # Calculate the training accuracy at each epoch
            stat = []
            stat += ([ll in best for ll, best in zip(lab, np.argsort(sc, axis=1)[:, -1:])])
            stat = np.average(stat)
            logger.info('Training accuracy at epoch {}: {}'.format(epoch, stat))

            # reduce the learning rate
            if epoch > 0 and ((epoch + 1) % self.lr_reduce) == 0:
                lr /= self.lr_factor
                saver.save(sess, os.path.join(weights_folder, 'reg_model'), global_step=epoch)

        # Extract weights
        self.save_weights = sess.run([variables_graph[i] for i in range(len(variables_graph))])

        # create inference model file
        dataset_utils.save_for_inference(weights_folder, self.save_weights,
                                         self.network, self.nb_classes,
                                         self.class_names, self.res_blocks,
                                 "features")
        # Save the entire incremental learning object
        with open(os.path.join(self.current_save_dir, "IncStone_algo.pickle"),
                  "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        logger.info("Finished training incremental classes. Saved algorithm to file")

    def incremental_train(self, inc_train_data, inc_classes, inc_cls_names, inc_exemplar_data=None):
        """Performs incremental training.

        Args:
            inc_train_data: filenames of the tfRecords used for training
            inc_classes: int list; indices of the classes used for incremental training.
            inc_cls_names: str list; names of the incremental classes
            inc_exemplar_data: if external data is given, otherwise the exemplars saved in the
                               object are used
        """
        self.all_classes.append(inc_classes)
        self.class_names.append(inc_cls_names)
        self.nb_classes += len(inc_classes)
        self.current_save_dir = os.path.join(self.results_path,
                                             "incremental_{}".format(len(self.all_classes) - 1))
        if not os.path.isdir(self.current_save_dir):
            os.makedirs(self.current_save_dir)

        weights_folder = os.path.join(self.current_save_dir, "weights")
        if not os.path.exists(weights_folder):
            os.makedirs(weights_folder)

        # include the exemplars from previous classes
        self.exemplars_batches = np.ceil(
            self.nb_protos_per_class * self.nb_classes / self.batch_size)
        total_batches = self.num_batches_inc + self.exemplars_batches

        if inc_exemplar_data is not None:
            dataset_inc, dataset_full = dataset_utils.dataset_tf_records(
                [inc_train_data, inc_exemplar_data],
                self.batch_size,
                [self.num_batches_inc, total_batches],
                # batches for new data, batches for new data +
                # batches to account for the exemplar data
                "inc_train")
        else:
            dataset_inc, dataset_full = dataset_utils.dataset_tf_records(
                [inc_train_data, self.exemplar_filenames],
                self.batch_size,
                [self.num_batches_inc, total_batches],
                "inc_train")

        iter_train, im_batch_train, lab_batch_train, lab_one_hot_train = dataset_utils.make_iterator(
            dataset_full, self.nb_classes)
        im_batch_train = tf.identity(im_batch_train, name="input")

        # Distillation
        variables_graph, variables_graph2, scores, scores_stored = dataset_utils.prepare_networks_train(
            self.network, im_batch_train, self.nb_classes, self.res_blocks)

        # Copying the network to use its predictions as ground truth labels
        op_assign = [(variables_graph2[i]).assign(variables_graph[i]) for i in
                     range(len(variables_graph))]

        # Define the objective for the neural network : 1 vs all cross_entropy + distillation
        scores = tf.concat(scores, 0)
        scores_stored = tf.concat(scores_stored, 0)
        scores = tf.identity(scores, name="output")
        old_cl = list(chain(*self.all_classes[:-1]))
        new_cl = self.all_classes[-1]
        label_old_classes = tf.sigmoid(tf.stack([scores_stored[:, i] for i in old_cl], axis=1))
        label_new_classes = tf.stack([lab_one_hot_train[:, i] for i in new_cl], axis=1)
        pred_old_classes = tf.stack([scores[:, i] for i in old_cl], axis=1)
        pred_new_classes = tf.stack([scores[:, i] for i in new_cl], axis=1)
        l2_reg = self.weight_decay * tf.reduce_sum(
            tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='Net'))
        loss_classif = tf.reduce_mean(tf.concat([tf.nn.sigmoid_cross_entropy_with_logits(
            labels=label_old_classes, logits=pred_old_classes),
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=label_new_classes,
                logits=pred_new_classes)], 1))
        loss = loss_classif + l2_reg
        learning_rate = tf.placeholder(tf.float32, shape=[])
        opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
        train_step = opt.minimize(loss, var_list=variables_graph)
        saver = tf.train.Saver()

        # Run the learning phase
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        lr = self.lr_old
        loss_batch = []

        b_temp = tf.get_variable("b_temp",
                                 shape=np.shape(self.save_weights[-1])[:-1] + (len(inc_classes),),
                                 initializer=tf.contrib.layers.xavier_initializer())
        self.save_weights[-1] = tf.concat([self.save_weights[-1], b_temp], axis=-1)

        W_temp = tf.get_variable("W_temp",
                                 shape=np.shape(self.save_weights[-2])[:-1] + (len(inc_classes),),
                                 initializer=tf.contrib.layers.xavier_initializer())
        self.save_weights[-2] = tf.concat([self.save_weights[-2], W_temp], axis=-1)
        sess.run(b_temp.initializer)
        sess.run(W_temp.initializer)

        # Run the loading of the weights for the learning network
        sess.run([(variables_graph[i]).assign(self.save_weights[i]) for i in
                  range(len(self.save_weights))])

        # Assign the weights of the learning network to the copy network
        sess.run(op_assign)
        if self.inc_chkpt is not None:
            saver.restore(sess, self.inc_chkpt)

        logger.info("********************************")
        logger.info("iCaRL incremental training on classes {} over {} epochs"
                    "".format(inc_classes, self.inc_epochs))
        logger.info("********************************")

        for epoch in range(self.inc_epochs):
            logger.info('Epoch {}'.format(epoch))
            sess.run(iter_train.initializer)
            np.save(os.path.join(self.current_save_dir, "biases_epoch_{}".format(epoch)),
                    variables_graph[-1].eval(session=sess))

            np.save(os.path.join(self.current_save_dir, "weights_epoch_{}".format(epoch)),
                    variables_graph[-2].eval(session=sess))

            batch = 0
            while True:
                try:
                    # begin = time.time()
                    loss_train, _, sc, lab = sess.run(
                        [loss_classif, train_step, scores, lab_batch_train],
                        feed_dict={learning_rate: lr})
                    loss_batch.append(loss_train)
                    if len(loss_batch) == min(self.num_batches_inc, 100):
                        logger.info("Training error: {}".format(np.mean(loss_batch)))
                        loss_batch = []
                    # end = time.time()
                    # logger.debug("Took {} second to train on one batch".format(end - begin))
                    batch += 1
                except tf.errors.OutOfRangeError:
                    break

            # Calculate the training accuracy at each epoch
            stat = []
            stat += ([ll in best for ll, best in zip(lab, np.argsort(sc, axis=1)[:, -1:])])
            stat = np.average(stat)
            logger.info('Training accuracy at epoch {}: {}'.format(epoch, stat))

            # reduce the learning rate
            if epoch > 0 and ((epoch + 1) % self.lr_reduce) == 0:
                lr /= self.lr_factor
                saver.save(sess, os.path.join(weights_folder, 'incremental_model'),
                           global_step=epoch)

        # Extract weights
        self.save_weights = sess.run([variables_graph[i] for i in range(len(variables_graph))])

        # Save exemplars for the incremental training data
        self.exemplars_management(dataset_inc, sess)
        sess.close()
        tf.reset_default_graph()

        # recalculate the means of the exemplars using the newly trained network
        self.recalculate_exemplars_means()

        # create inference model file
        dataset_utils.save_for_inference(weights_folder, self.save_weights,
                                         self.network, self.nb_classes,
                                         self.class_names, self.res_blocks,
                                 "features")

        class_exemplar_means_file = os.path.join(self.current_save_dir, "means.txt")
        np.savetxt(class_exemplar_means_file, X=self.class_exemplar_means,
                   delimiter=" ", newline="\n", fmt="%.18f")

        # Save the entire incremental learning object
        with open(os.path.join(self.current_save_dir, "IncStone_algo.pickle"),
                  "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        logger.info("Finished training incremental classes. Saved algorithm to file")

    def recalculate_exemplars_means(self):
        logger.info('Exemplars means recalculation starting ...')
        dataset_exemplars = dataset_utils.dataset_tf_records(self.exemplar_filenames,
                                                             self.batch_size,
                                                             self.exemplars_batches,
                                                     "test")
        classes = list(chain(*self.all_classes[:-1]))
        num_classes = len(classes)
        iter, im_batch, lab_batch, lab_one_hot = dataset_utils.make_iterator(dataset_exemplars, num_classes)
        inits, _, op_feature_map = dataset_utils.prepare_network_test(self.network, im_batch,
                                                                      self.nb_classes,
                                                                      self.save_weights, self.res_blocks)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(inits)
        sess.run(iter.initializer)

        feature_maps_size = op_feature_map.get_shape()[3].value
        class_means = np.zeros((num_classes, feature_maps_size))
        class_summed_features = np.zeros((num_classes, feature_maps_size))
        running_image_count = np.zeros((num_classes), dtype=int)

        # Calculate the feature mapped means for the whole dataset
        while True:
            try:
                _, labels_inc, _, feature_mapped_images = dataset_utils.load_in_feature_space(
                    im_batch, lab_batch, lab_one_hot, op_feature_map, sess)
                for idx, cls in enumerate(classes):
                    cls_idx = np.where(labels_inc == cls)[0]
                    if len(cls_idx) > 0:
                        cls_feature_maps = feature_mapped_images[:, cls_idx]
                        running_image_count[idx] += len(cls_idx)
                        class_summed_features[idx] += np.sum(cls_feature_maps, axis=1)
                        class_means[idx] = (class_summed_features[idx]
                                            / running_image_count[idx])
            except tf.errors.OutOfRangeError:
                break
        for idx in range(len(classes)):
            class_means[idx] /= np.linalg.norm(class_means[idx])

        sess.close()
        tf.reset_default_graph()
        self.class_exemplar_means[:num_classes] = class_means

    def exemplars_in_memory(self, dataset_inc, sess):
        """Given training images dataset, it saves a list of exemplars to be used for incremental
        learning. iCARL-identical management method.

        Args:
            dataset_full: tensorflow dataset
            curr_classes: indices of the classes for which the exemplars are saved
            sess: tensorflow session
        """
        curr_classes = self.all_classes[-1]
        num_inc_classes = len(curr_classes)

        # calculate exemplars for new incremental classes
        logger.info('Exemplars selection starting ...')
        iter_inc, im_batch_inc, lab_batch_inc, lab_one_hot_inc = dataset_utils.make_iterator(
            dataset_inc, self.nb_classes)

        inits, _, op_feature_map = dataset_utils.prepare_network_test(self.network,
                                                                      im_batch_inc,
                                                                      self.nb_classes,
                                                                      self.save_weights,
                                                                      self.res_blocks)
        sess.run(inits)
        sess.run(iter_inc.initializer)

        images = []
        feature_maps = []
        labels = []
        while True:
            try:
                images_batch, labels_inc, _, feature_mapped_images = dataset_utils.load_in_feature_space(
                    im_batch_inc, lab_batch_inc, lab_one_hot_inc, op_feature_map, sess)
                feature_maps.append(feature_mapped_images.T)
                images.append(images_batch)
                labels.extend(labels_inc)
            except tf.errors.OutOfRangeError:
                break
        images = np.concatenate(images, axis=0)
        feature_maps = np.concatenate(feature_maps, axis=0).T
        labels = np.asarray(labels)

        # calculate how many images there are for each class
        class_indices = []
        for curr_cls_idx, curr_cls in enumerate(curr_classes):
            class_indices.append(np.where(labels == curr_cls)[0])

        feature_maps_size = op_feature_map.get_shape()[3].value
        class_exemplars_list = [[] for _ in range(num_inc_classes)]
        class_exemplars_maps_list = [[] for _ in range(num_inc_classes)]
        class_exemplars_means_list = [[] for _ in range(num_inc_classes)]
        feature_maps_size = op_feature_map.get_shape()[3].value
        class_means_full_list = np.zeros((num_inc_classes, feature_maps_size))
        nb_samples = [len(cls) for cls in class_indices]
        self.nb_protos_per_class = min(self.nb_protos_per_class, np.amin(len(nb_samples)))

        for curr_cls_idx, curr_cls in enumerate(curr_classes):
            idx_current_class = class_indices[curr_cls_idx]
            mask = np.ones((len(idx_current_class)))
            curr_cls_feature_maps = feature_maps[:, idx_current_class]
            curr_cls_images = images[idx_current_class]
            class_mean = np.mean(curr_cls_feature_maps, axis=1)
            class_means_full = class_mean / np.linalg.norm(class_mean)
            class_exemplars = np.zeros((self.nb_protos_per_class,) + np.shape(curr_cls_images)[1:])
            class_exemplars_maps = np.zeros((self.nb_protos_per_class,
                                             np.shape(curr_cls_feature_maps)[0]))

            for class_exemplars_idx in range(self.nb_protos_per_class):
                distances = []
                temp_feature_maps_transpose = np.transpose(curr_cls_feature_maps)
                if class_exemplars_idx > 0:
                    mean_features = np.transpose([np.mean(np.concatenate(
                        [class_exemplars_maps[:class_exemplars_idx],
                         np.expand_dims(feature_map, axis=0)]), axis=0) for feature_map in
                        temp_feature_maps_transpose])
                else:
                    mean_features = curr_cls_feature_maps
                ind_max = np.nanargmax(np.dot(class_mean, mean_features) * mask)
                mask[ind_max] = np.NaN
                class_exemplars[class_exemplars_idx] = curr_cls_images[ind_max]
                class_exemplars_maps[class_exemplars_idx] = curr_cls_feature_maps[:, ind_max]

            class_exemplars_list[curr_cls_idx] = [class_exemplars[i] for i in
                                                  range(self.nb_protos_per_class)]
            class_exemplars_maps_list[curr_cls_idx] = [class_exemplars_maps[i] for i in
                                                       range(self.nb_protos_per_class)]
            class_means_full_list[curr_cls_idx] = class_means_full
            # Compute the mean of exemplar maps for each class seen so far
            exemplar_maps_means = np.mean(class_exemplars_maps, axis=0)
            exemplar_maps_means = (exemplar_maps_means
                                   / np.linalg.norm(exemplar_maps_means))
            class_exemplars_means_list[curr_cls_idx] = exemplar_maps_means

        exemplars_labels_list = [[cls] * self.nb_protos_per_class for cls in curr_classes]
        self.class_full_means.extend(class_means_full_list)
        self.class_exemplar_means.extend(class_exemplars_means_list)
        self.exemplar_filenames.extend(dataset_utils.exemplars_to_tfrecord(class_exemplars_list,
                                                                           exemplars_labels_list,
                                                                           curr_classes,
                                                                           self.current_save_dir))

    def exemplars_management(self, dataset_inc, sess):
        """Given training images dataset, it saves a list of exemplars to be used for incremental
        learning. Slightly modified iCARL management method.

        Args:
            dataset_train: tensorflow dataset
            curr_classes: indices of the classes for which the exemplars are saved
            sess: tensorflow session
        """
        logger.info('Exemplars selection starting ...')
        curr_classes = self.all_classes[-1]
        num_inc_classes = len(curr_classes)
        iter_inc, im_batch_inc, lab_batch_inc, lab_one_hot_inc = dataset_utils.make_iterator(
            dataset_inc, self.nb_classes)

        inits, _, op_feature_map = dataset_utils.prepare_network_test(self.network,
                                                                      im_batch_inc,
                                                                      self.nb_classes,
                                                                      self.save_weights,
                                                                      self.res_blocks)
        sess.run(inits)
        sess.run(iter_inc.initializer)
        # Means for the feature mapped images of each class (for all the training images for a
        #  specific class)
        feature_maps_size = op_feature_map.get_shape()[3].value
        class_means_full = np.zeros((num_inc_classes, feature_maps_size))
        class_summed_features = np.zeros((num_inc_classes, feature_maps_size))
        running_image_count = np.zeros((num_inc_classes), dtype=int)

        # Means of the feature mapped images for the chosen exemplars
        class_exemplars = [[] for _ in range(num_inc_classes)]
        class_exemplars_maps = [[] for _ in range(num_inc_classes)]
        class_exemplars_dot_product = [[] for _ in range(num_inc_classes)]

        # Calculate the feature mapped means for the whole dataset
        while True:
            try:
                _, labels_inc, _, feature_mapped_images = dataset_utils.load_in_feature_space(
                    im_batch_inc, lab_batch_inc, lab_one_hot_inc, op_feature_map, sess)
                for idx, cls in enumerate(curr_classes):
                    cls_idx = np.where(labels_inc == cls)[0]
                    if len(cls_idx) > 0:
                        cls_feature_maps = feature_mapped_images[:, cls_idx]
                        running_image_count[idx] += len(cls_idx)
                        class_summed_features[idx] += np.sum(cls_feature_maps, axis=1)
                        class_means_full[idx] = (class_summed_features[idx]
                                                 / running_image_count[idx])
            except tf.errors.OutOfRangeError:
                break
        for idx in range(len(curr_classes)):
            pass
        class_means_full[idx] /= np.linalg.norm(class_means_full[idx])
        self.nb_protos_per_class = min(self.nb_protos_per_class, np.amin(running_image_count))
        # Save only a limited number of exemplars which best approaches the class feature mapped
        # mean
        begin = time.time()
        sess.run(iter_inc.initializer)
        while True:
            try:
                images_inc, labels_inc, _, feature_mapped_images = dataset_utils.load_in_feature_space(
                    im_batch_inc, lab_batch_inc, lab_one_hot_inc, op_feature_map, sess)
                for idx_e, element in enumerate(zip(feature_mapped_images.T, labels_inc)):
                    feature_map = element[0]
                    label = element[1]
                    cls_idx = curr_classes.index(label)
                    mean_distance_element = np.dot(feature_map, class_means_full[cls_idx])
                    insert_index = bisect.bisect(class_exemplars_dot_product[cls_idx],
                                                 mean_distance_element)
                    class_exemplars_dot_product[cls_idx].insert(insert_index,
                                                                mean_distance_element)
                    class_exemplars[cls_idx].insert(insert_index,
                                                    images_inc[idx_e])
                    class_exemplars_maps[cls_idx].insert(insert_index,
                                                         feature_map)
                class_exemplars = [arr[-self.nb_protos_per_class:] for arr in class_exemplars]
                class_exemplars_dot_product = [arr[-self.nb_protos_per_class:]
                                               for arr in class_exemplars_dot_product]
                class_exemplars_maps = [arr[-self.nb_protos_per_class:] for arr in
                                        class_exemplars_maps]
            except tf.errors.OutOfRangeError:
                break

        total = time.time() - begin
        logger.debug("It took {} sec to compute {}"
                     " class exemplars".format(total,
                                               len(class_exemplars[0]) * len(class_exemplars)))
        exemplars_labels = [[cls] * self.nb_protos_per_class for cls in curr_classes]
        self.class_full_means.extend(class_means_full)
        exemplar_maps_means = np.mean(class_exemplars_maps, axis=1)
        exemplar_maps_means = (exemplar_maps_means.T
                               / np.linalg.norm(exemplar_maps_means, axis=1)).T
        self.class_exemplar_means.extend(exemplar_maps_means)
        self.exemplar_filenames.extend(dataset_utils.exemplars_to_tfrecord(class_exemplars,
                                                                           exemplars_labels,
                                                                           self.all_classes[-1],
                                                                           self.current_save_dir))

    def evaluate(self, filenames_test, eval_classes, net_type):
        """Perform evaluation of the algorithm. Saves the final accuracies to file.

        Args:
            filenames_test: filenames of the tfRecords used for testing
            eval_classes: the indices of classes used for testing
            type: whether to evaluate in icarl style, or only the network output
        """
        logger.info("Starting evaluation for classes {}".format(eval_classes))
        accuracy_list = []
        if net_type == "icarl":
            exemplar_maps_means = np.asarray(self.class_exemplar_means)
        dataset_test = dataset_utils.dataset_tf_records(filenames_test, self.batch_size,
                                                        self.num_batches_eval, "test")
        iter_test, im_batch_test, lab_batch_test, lab_one_hot_test = dataset_utils.make_iterator(
            dataset_test, self.nb_classes)
        inits, scores, op_feature_map = dataset_utils.prepare_network_test(self.network,
                                                                           im_batch_test,
                                                                           self.nb_classes,
                                                                           self.save_weights,
                                                                           self.res_blocks)
        sess = tf.Session(config=config)
        sess.run(inits)
        sess.run(iter_test.initializer)

        stat_hb1 = np.zeros((len(eval_classes), 2), dtype=np.int32)
        if net_type == "icarl":
            stat_icarl = np.zeros((len(eval_classes), 2), dtype=np.int32)

        def balance_acc(stats, labels, preds):
            present_labels = np.unique(labels)
            for lab in present_labels:
                location = eval_classes.index(lab)
                present_idx = np.where(labels == lab)[0]
                pos = np.sum(preds[present_idx] == lab)
                neg = len(present_idx) - pos
                stats[location][0] += pos
                stats[location][1] += neg

        while True:
            try:
                sc, labels, labels_one_hot, feat_map_tmp = sess.run(
                    [scores, lab_batch_test, lab_one_hot_test, op_feature_map])
                balance_acc(stat_hb1, labels, np.argsort(sc, axis=1)[:, -1])

                if net_type == "icarl":
                    feature_maps_batch = feat_map_tmp[:, 0, 0, :]
                    pred_inter = (feature_maps_batch.T) / np.linalg.norm(feature_maps_batch.T,
                                                                         axis=0)
                    sqd_icarl = -cdist(exemplar_maps_means, pred_inter.T, 'sqeuclidean').T
                    icarl_labels = np.argsort(sqd_icarl, axis=1)[:, -1]
                    balance_acc(stat_icarl, labels, icarl_labels)
            except tf.errors.OutOfRangeError:
                break
        logger.info('Classes: {}'.format(eval_classes))

        hybrid_acc = np.mean(stat_hb1[:, 0] / np.sum(stat_hb1, axis=1))
        logger.info('Hybrid 1 accuracy: {}'.format(hybrid_acc))
        accuracy_list.append(hybrid_acc)

        if net_type == "icarl":
            icarl_acc = np.mean(stat_icarl[:, 0] / np.sum(stat_icarl, axis=1))
            logger.info('iCaRL accuracy: {}'.format(icarl_acc))
            accuracy_list.append(icarl_acc)

        sess.close()
        tf.reset_default_graph()
        with open(
                os.path.join(self.current_save_dir, "accuracy_classes_{}.txt".format(eval_classes)),
                "wb") as f:
            np.savetxt(f, accuracy_list)
