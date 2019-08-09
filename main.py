from __future__ import print_function
import numpy as np
import pickle
from tools import dataset_utils
import argparse
import os
import time
import re
import copy
import pandas as pd
import random as rn
import tensorflow as tf
from incRoshambo import IncRoshambo
from tools.misc_utils import make_results_dir, initialize_logger, experiment_details


def load_pickled_algo(args, logger):
    """
    Args:
        args: all the arguments given from console, used to set the right params to loaded object
        logger: logger object

    Returns:
        the loaded algorithm to use from training and testing

    """

    def load_dir_exemplars(path):
        files = os.listdir(path)
        exemplars_dirs = [d for d in files if "exemplars_" in d]
        assert len(exemplars_dirs) == 1, "The folder should only contain one subfolder with saved " \
                                         "exemplars"
        exemplars_dir = exemplars_dirs[0]
        exemplars_files = os.listdir(os.path.join(path, exemplars_dir))
        return [os.path.join(path, exemplars_dir, file) for file in exemplars_files if ".tfrecords"
                in file]

    def sort_filenames(filename_list):
        convert = lambda text: float(text) if text.isdigit() else text
        alphanum = lambda key: [convert(c) for c in re.split('([-+]?[0-9]*\.?[0-9]*)', key)]
        filename_list.sort(key=alphanum)
        return filename_list

    assert args.algo_pickle_path is not None, "You have to give the path to a trained base network"
    algo_path = os.path.join(args.algo_pickle_path, args.algo_pickle_type, "IncStone_algo.pickle")
    exemplars_paths = []
    if args.algo_pickle_type == "base":
        exemplars_paths.extend(sort_filenames(load_dir_exemplars(os.path.join(args.algo_pickle_path,
                                                                              "base"))))
    elif "incremental" in args.algo_pickle_type:
        inc_index = int(args.algo_pickle_type[-1])
        exemplars_paths.extend(sort_filenames(load_dir_exemplars(os.path.join(args.algo_pickle_path,
                                                                              "base"))))
        for i in range(1, inc_index + 1):
            inc_path = os.path.join(args.algo_pickle_path, "incremental_{}".format(i))
            exemplars_paths.extend(sort_filenames(load_dir_exemplars(inc_path)))
    else:
        logger.error("You have given a wrong pickle network type. Valid are 'base' or "
                     "'incremental_idx'")
    algo = pickle.load(open(algo_path, 'rb'))
    algo.results_path = args.results_path
    algo.nb_protos_per_class = args.exemplars_memory
    algo.current_save_dir = os.path.join(args.results_path, "base")

    if not os.path.exists(algo.current_save_dir):
        os.makedirs(algo.current_save_dir)

    algo.exemplar_filenames = exemplars_paths
    algo.inc_epochs = args.inc_epochs
    algo.base_epochs = args.base_epochs
    algo.num_batches_inc = args.num_batches_inc
    algo.num_batches_eval = args.num_batches_eval
    algo.num_batches_base = args.num_batches_base
    return algo


def base_training_phase(args, train_recs, test_recs, cls_indices, cls_names):
    algo = IncRoshambo(args)
    algo.base_train(train_recs, cls_indices, cls_names)
    algo.evaluate(test_recs, cls_indices, "icarl")
    return algo


def icarl_inc_training_phase(algo, inc_train, inc_test, base_test_tfrecs, inc_class_idx,
                             base_class_idx, inc_class_names):
    algo.incremental_train(inc_train, inc_class_idx, inc_class_names)
    algo.evaluate(base_test_tfrecs, base_class_idx, "icarl")
    algo.evaluate(inc_test, inc_class_idx, "icarl")
    return algo


def regular_inc_training_phase(algo, inc_train, inc_test, base_test_tfrecs, inc_class_idx,
                               base_class_idx, inc_class_names):
    algo.regular_train(inc_train, inc_class_idx, inc_class_names)
    algo.evaluate(base_test_tfrecs, base_class_idx, "regular")
    algo.evaluate(inc_test, inc_class_idx, "regular")
    return algo


def offline_pipeline(args, logger):
    dataset = pd.read_csv(
        os.path.join(args.data_dir, "dataset_description.csv"))
    base_class_idx = list(range(args.base_classes))
    classes = dataset.iloc[:-1]
    base_info = dataset.iloc[base_class_idx]

    inc_indices = np.arange(args.base_classes, len(classes))
    inc_iters = np.arange(0, len(classes) - args.base_classes, 2)
    np.random.shuffle(inc_indices)

    base_train_tfrecs, base_test_tfrecs, base_class_names = dataset_utils.save_npy_to_tfrecord(
        base_info,
        base_class_idx,
        args.data_dir,
        args.results_path)

    if args.base_knowledge == "big":
        algo = load_pickled_algo(args, logger)
        icarl_algo = copy.deepcopy(algo)
        icarl_algo.evaluate(base_test_tfrecs, base_class_idx, "regular")
    else:
        icarl_algo = base_training_phase(args, base_train_tfrecs, base_test_tfrecs,
                                         base_class_idx.copy(),
                                         base_class_names)

    base_offset = args.base_classes

    for iter in inc_iters:
        idx_slice = np.array([iter, iter + 1])
        symbol_idx = [inc_indices[iter], inc_indices[iter + 1]]
        inc_class_idx = list(idx_slice + base_offset)
        inc_info = classes.iloc[symbol_idx]
        inc_train_tfrecs, inc_test_tfrecs, inc_class_names = dataset_utils.save_npy_to_tfrecord(
            inc_info,
            inc_class_idx.copy(),
            args.data_dir,
            args.results_path)

        # icarl training and testing
        logger.info("Training with distillation and nearest-mean on {} after having"
                    " trained on the base classes.".format(inc_class_names))
        beginning = time.time()
        icarl_algo = icarl_inc_training_phase(icarl_algo, inc_train_tfrecs, inc_test_tfrecs,
                                              base_test_tfrecs,
                                              inc_class_idx.copy(),
                                              base_class_idx.copy(),
                                              inc_class_names)
        ending = time.time()
        elapsed = ending - beginning
        np.save(os.path.join(args.results_path, "incremental_learning_duration.txt"),
                np.asarray(elapsed))

        base_test_tfrecs.extend(inc_test_tfrecs)
        base_class_idx.extend(inc_class_idx)


def parse_args():
    """Parses arguments specified on the command-line
    """
    argparser = argparse.ArgumentParser('Train and evaluate Roshambo iCarl')
    argparser.add_argument('--image_dims',
                           help="the dimensions of the images we are working with",
                           default=(64, 64, 1))
    argparser.add_argument('--console_print',
                           help="If set to true, it prints logger info to console.",
                           type=bool, default=False)
    argparser.add_argument('--batch_size', type=int,
                           help="The number of images to process at the same time",
                           default=512)
    argparser.add_argument('--num_batches_base', type=int,
                           help="Over how many batches to train. Set to -1 for all available data.",
                           default=-1)
    argparser.add_argument('--num_batches_inc', type=int,
                           help="Over how many batches to train. Set to -1 for all available data.",
                           default=-1)
    argparser.add_argument('--num_batches_eval', type=int,
                           help="Over how many batches to evaluate. "
                                "Set to -1 for all available data.",
                           default=-1)
    argparser.add_argument('--network', type=str,
                           help="The network type to use for training and testing. Options:"
                                "[RoshamboNet, ResNet]", default="ResNet")
    argparser.add_argument('--res_blocks', type=int,
                           help="If the network is ResNet, specify the number of residual blocks,"
                                "otherwise leave as None",
                           default=2)
    argparser.add_argument('--base_epochs', type=int,
                           help="The number of epochs over which to base train the network",
                           default=100)
    argparser.add_argument('--inc_epochs', type=int,
                           help="The number of epochs for incrementally training the network",
                           default=100)
    argparser.add_argument('--lr_old', type=float,
                           help="The initial learning rate of the network",
                           default=0.01)
    argparser.add_argument('--lr_factor', type=float,
                           help="By how much the learning rate decreases over epochs",
                           default=1.0)
    argparser.add_argument('--weight_decay', type=float,
                           help="By how much the weights decay",
                           default=1e-5)
    argparser.add_argument('--seed', help="Random seed to make experiments reproducible",
                           type=int, default=1)
    argparser.add_argument('--lr_reduce', type=int,
                           help="After how many epochs the learning rate decreases",
                           default=10)
    argparser.add_argument('--exemplars_memory', type=int,
                           help="How many exemplars to store for each class if the reduce flag is"
                                "set to False, or how many to have in total for all classes,"
                                "if the reduce flag is set to True",
                           default=4000)
    argparser.add_argument('--save_path',
                           help="Path for results saving",
                           default="./results")
    argparser.add_argument('--data_dir',
                           help="Path where data is stored",
                           default="/mnt/data/datasets/roshambo/event_based_icarl")
    argparser.add_argument('--base_chkpt',
                           help="Path where the saved weights for the base net are",
                           default=None)
    argparser.add_argument('--inc_chkpt',
                           help="Path where the saved weights for the inc net are",
                           default=None)
    argparser.add_argument('--algo_pickle_path',
                           help="Path to where the whole incRoshambo class instance you want to "
                                "load has been pickled",
                           default="./trained_base")
    argparser.add_argument('--algo_pickle_type',
                           help="whether you want to load the base network or an already"
                                "incrementally trained network",
                           default="base")
    argparser.add_argument('--base_knowledge',
                           help="if you want to start from a pretrained network with vast "
                                "base knowledge, type big. otherwise type small",
                           default="big")
    argparser.add_argument('--base_classes', type=int,
                           help="how many classes to use as base knowledge",
                           default=4)
    return argparser.parse_args()


def main():
    args = parse_args()

    # make the experiment deterministic
    np.random.seed(args.seed)
    rn.seed(args.seed)
    tf.set_random_seed(args.seed)
    os.environ["PYTHONSEED"] = str(args.seed)

    args.results_path = make_results_dir(args.save_path, args.inc_epochs, args.seed,
                                         args.exemplars_memory)
    logger = initialize_logger(args.results_path, args.console_print)
    experiment_details(args)
    offline_pipeline(args, logger)


if __name__ == '__main__':
    main()
