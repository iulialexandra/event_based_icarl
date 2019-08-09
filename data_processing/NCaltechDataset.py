import os
import numpy as np
import csv
from random import shuffle
from tools.dataset_utils import bin_to_frame_list, augment_dataset
import argparse


class NCaltechDataset():
    def __init__(self, args):
        self.rotation_range = args.rotation_range
        self.width_shift_range = args.width_shift_range
        self.height_shift_range = args.height_shift_range
        self.brightness_range = args.brightness_range
        self.shear_range = args.shear_range
        self.zoom_range = args.zoom_range
        self.channel_shift_range = args.channel_shift_range
        self.fill_mode = args.fill_mode
        self.cval = args.cval
        self.horizontal_flip = args.horizontal_flip
        self.vertical_flip = args.vertical_flip
        self.dataset_path = args.input_path
        self.train_augment = args.train_augment
        self.test_augment = args.test_augment
        self.save_path = args.save_path
        self.resize_scale = args.resize_scale
        self.max_frames_per_vid = args.max_frames_per_vid

    def save_symbols_to_npy(self, data_path):
        path_contents = os.listdir(data_path)
        symbols = [symbol for symbol in path_contents if
                   os.path.isdir(os.path.join(data_path, symbol))]
        paths = [os.path.join(data_path, symbol) for symbol in symbols]

        dataset_description = open(os.path.join(self.save_path, 'dataset_description.csv'),
                                   'w')
        with dataset_description:
            fields = ["symbol", "train_file", "test_file", "train_num_samples", "test_num_samples"]
            csv_writer = csv.DictWriter(dataset_description, fieldnames=fields)
            csv_writer.writeheader()

            for s, symbol in enumerate(symbols):
                print("Loading symbol: " + symbol)
                symbol_path = paths[s]
                symbol_data = []
                symbol_recs = [rec for rec in os.listdir(symbol_path) if ".bin" in rec]
                for recording in symbol_recs:
                    rec_path = os.path.join(symbol_path, recording)
                    frame_list = bin_to_frame_list(rec_path,
                                                   (64, 64))
                    if frame_list is not None:
                        symbol_data.extend(frame_list)

                shuffle(symbol_data)
                split = int(0.8 * len(symbol_data))
                train_data = symbol_data[:split]
                test_data = symbol_data[split:]

                train_data = augment_dataset(train_data, self.train_augment,
                                             rotation_range=self.rotation_range,
                                             width_shift_range=self.width_shift_range,
                                             height_shift_range=self.height_shift_range,
                                             brightness_range=self.brightness_range,
                                             shear_range=self.shear_range,
                                             zoom_range=self.zoom_range,
                                             channel_shift_range=self.channel_shift_range,
                                             fill_mode=self.fill_mode,
                                             cval=self.cval,
                                             horizontal_flip=self.horizontal_flip,
                                             vertical_flip=self.vertical_flip)
                if self.test_augment > 0:
                    test_data = augment_dataset(test_data, self.test_augment,
                                                rotation_range=self.rotation_range,
                                                width_shift_range=self.width_shift_range,
                                                height_shift_range=self.height_shift_range,
                                                brightness_range=self.brightness_range,
                                                shear_range=self.shear_range,
                                                zoom_range=self.zoom_range,
                                                channel_shift_range=self.channel_shift_range,
                                                fill_mode=self.fill_mode,
                                                cval=self.cval,
                                                horizontal_flip=self.horizontal_flip,
                                                vertical_flip=self.vertical_flip)

                train_data = np.asarray(train_data, dtype=np.uint8)
                test_data = np.asarray(test_data, dtype=np.uint8)

                train_path = "{}_train.npy".format(symbol)
                test_path = "{}_test.npy".format(symbol)

                np.save(os.path.join(self.save_path, train_path), train_data)
                np.save(os.path.join(self.save_path, test_path), test_data)

                csv_writer.writerow({"symbol": symbol, "train_file": train_path,
                                     "test_file": test_path,
                                     "train_num_samples": len(train_data),
                                     "test_num_samples": len(test_data)})


def main(args):
    loader = NCaltechDataset(args)
    loader.save_symbols_to_npy(args.input_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help="Path to recordings",
                        default="/mnt/data/datasets/N-Caltech101/Caltech101")
    parser.add_argument("--save_path", help="Path to saved dataset",
                        default="/mnt/data/datasets/N-Caltech101/event_based_icarl")
    parser.add_argument('--resize_scale', default=None)
    parser.add_argument('--max_frames_per_vid', default=-1)
    parser.add_argument('--rotation_range', default=15.)
    parser.add_argument('--width_shift_range', default=0.1)
    parser.add_argument('--height_shift_range', default=0.1)
    parser.add_argument('--brightness_range', default=None)
    parser.add_argument('--shear_range', default=0.1)
    parser.add_argument('--zoom_range', default=0.15)
    parser.add_argument('--channel_shift_range', default=0.15)
    parser.add_argument('--fill_mode', default='nearest')
    parser.add_argument('--cval', default=0.)
    parser.add_argument('--horizontal_flip', default=True)
    parser.add_argument('--vertical_flip', default=False)
    parser.add_argument('--data_format', default=None)
    parser.add_argument('--train_augment', default=5000)
    parser.add_argument('--test_augment', default=1000)
    args = parser.parse_args()
    main(args)
