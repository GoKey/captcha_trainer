#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import cv2
import sys
import random
import PIL.Image
import numpy as np
import tensorflow as tf
from pretreatment import preprocessing
from config import *

REGEX_MAP = {
    RunMode.Trains: TRAINS_REGEX,
    RunMode.Test: TEST_REGEX
}

_RANDOM_SEED = 0

if not os.path.exists(TFRECORDS_DIR):
    os.makedirs(TFRECORDS_DIR)


def _image(path):

    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        pil_image = PIL.Image.open(path).convert("RGB")
        im = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2GRAY)
    im = preprocessing(im, BINARYZATION, SMOOTH, BLUR)
    im = cv2.resize(im, (IMAGE_WIDTH, IMAGE_HEIGHT))
    return im


def _dataset_exists(dataset_dir):
    for split_name in TFRECORDS_NAME_MAP.values():
        output_filename = os.path.join(dataset_dir, split_name + '.tfrecords')
        if not tf.gfile.Exists(output_filename):
            return False
    return True


def _get_all_files(dataset_dir):
    file_list = []
    for filename in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, filename)
        file_list.append(path)
    return file_list


def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def image_to_tfrecords(image_data, label):
    return tf.train.Example(features=tf.train.Features(feature={
        'image': bytes_feature(image_data),
        'label': bytes_feature(label),
    }))


def _convert_dataset(file_list, mode):

    output_filename = os.path.join(TFRECORDS_DIR, TFRECORDS_NAME_MAP[mode] + '.tfrecords')
    with tf.python_io.TFRecordWriter(output_filename) as writer:
        for i, file_name in enumerate(file_list):
            try:
                sys.stdout.write('\r>> Converting image %d/%d ' % (i + 1, len(file_list)))
                sys.stdout.flush()
                image_data = _image(file_name)
                image_data = image_data.tobytes()
                labels = re.search(REGEX_MAP[mode], file_name.split(PATH_SPLIT)[-1]).group()
                labels = labels.encode('utf-8')

                example = image_to_tfrecords(image_data, labels)
                writer.write(example.SerializeToString())

            except IOError as e:
                print('could not read:', file_list[1])
                print('error:', e)
                print('skip it \n')
    sys.stdout.write('\n')
    sys.stdout.flush()


if __name__ == '__main__':

    if _dataset_exists(TFRECORDS_DIR):
        print('Exists!')
    else:
        test_dataset = _get_all_files(TEST_PATH)
        trains_dataset = _get_all_files(TRAINS_PATH)
        random.seed(_RANDOM_SEED)
        random.shuffle(test_dataset)
        random.shuffle(trains_dataset)
        _convert_dataset(test_dataset, mode=RunMode.Test)
        _convert_dataset(trains_dataset, mode=RunMode.Trains)
        print("Done!")
