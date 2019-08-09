from os import listdir
from os.path import isfile, join
import sys
import json
import datetime
import logging
import os
from random import shuffle
from networks import ResNet, RoshamboNet


def sweep_directories(directory_list, token=None):
    """Given a list of directories, it returns the files inside"""
    files = []
    if type(directory_list) != list:
        directory_list = [os.path.join(directory_list, direc) for direc in os.listdir(directory_list)]
    for directory in directory_list:
        if "symbol" in directory:
            for f in listdir(directory):
                file = join(directory, f)
                if (isfile(file) and token in file):
                    files.append(file)
    shuffle(files)
    return files


def extract_subdirs(main_dir, name_pattern):
    valid_subdirs = []
    for subdir in os.listdir(main_dir):
        full_subdir = os.path.join(main_dir, subdir)
        if os.path.isdir(full_subdir) and name_pattern in str(full_subdir):
            valid_subdirs.append(full_subdir)
    return valid_subdirs


def initialize_logger(output_dir, print_to_console):
    """initializes loggers for debug and error, prints to file

    Args:
        output_dir: the directory of the logger file
        print_to_console: flag, whether to print logging info to console
    """
    logger = logging.getLogger("roshambo_demo")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s",
                                  "%d-%m-%Y %H:%M:%S")
    # Setup console logging
    if print_to_console:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # Setup file logging
    fh = logging.FileHandler(os.path.join(output_dir, "log.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.propagate = False
    return logger


def make_results_dir(save_path, inc_epocs, seed, exemplars):
    """Makes one folder for the results using the current date and time
     and initializes the logger.
    """
    now = datetime.datetime.now()
    date = "{}_{}_{}-{}_{}_{}_{}".format(now.year, now.month, now.day, now.hour, now.minute,
                                         now.second, now.microsecond)
    results_path = os.path.join(save_path,
                                date + "seed_{}".format(seed) + "_{}_epochs".format(inc_epocs)
                                + "_{}_exemplars".format(exemplars))
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    # else:
    #     path_list = self.base_chkpt.split("/")[: -3]
    #     self.results_path = os.path.join("/", *path_list)
    return results_path


def experiment_details(args):
    """Writes the arguments to file to keep track of various experiments"""
    exp_path = os.path.join(args.results_path, "experiment_details.json")
    if not os.path.exists(exp_path):
        with open(exp_path, "w") as outfile:
            json.dump(vars(args), outfile)


def str_to_class(str):
    """Gets a class when given its name.

    Args:
        str: the name of the class to retrieve
    """
    return getattr(sys.modules[__name__], str)
