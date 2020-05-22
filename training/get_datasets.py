import numpy as np
import glob
import os

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from constants import DATA_DIR


def get_data_for_dataset(dataset_name, mode):
    # Implement this for each dataset.
    dataset = None
    if dataset_name == "imagenet_video":
        image_datadir = os.path.join(DATA_DIR, "Imagenet_Video", "provided_small")
        datadir = os.path.join(os.path.dirname(__file__), "datasets", "imagenet_video")
        gt = np.load(datadir + "/labels/" + mode + "/labels_small_boxes.npy")
        image_paths = [
            image_datadir + "/" + line[5:].strip() for line in open(datadir + "/labels/" + mode + "/image_names.txt")
        ]

        """
        image_datadir = os.path.join(
                DATA_DIR,
                'Imagenet_Video',
                'provided')
        datadir = os.path.join(
                os.path.dirname(__file__),
                'datasets',
                'imagenet_video')
        gt = np.load(datadir + '/labels/' + mode + '/labels.npy')
        image_paths = [image_datadir + '/' + line[5:].strip()
            for line in open(datadir + '/labels/' + mode + '/image_names.txt')]
        """

    elif dataset_name == "alov":
        image_datadir = os.path.join(DATA_DIR, "ALOV")
        datadir = os.path.join(os.path.dirname(__file__), "datasets", "alov")
        gt = np.load(datadir + "/labels/" + mode + "/labels.npy")
        image_paths = [
            image_datadir + "/" + line.strip() for line in open(datadir + "/labels/" + mode + "/image_names.txt")
        ]
    elif dataset_name == "caltech_pedestrian":
        image_datadir = os.path.join(DATA_DIR, "Caltech_Pedestrian", "dataset",)
        datadir = os.path.join(os.path.dirname(__file__), "datasets", "caltech_pedestrian")
        gt = np.load(datadir + "/labels/" + mode + "/labels.npy")
        image_paths = [
            image_datadir + "/" + line.strip() for line in open(datadir + "/labels/" + mode + "/image_names.txt")
        ]
    elif dataset_name == "kitti":
        image_datadir = os.path.join(DATA_DIR, "KITTI", "dataset",)
        datadir = os.path.join(os.path.dirname(__file__), "datasets", "kitti")
        gt = np.load(datadir + "/labels/" + mode + "/labels.npy")
        image_paths = [
            image_datadir + "/" + line.strip() for line in open(datadir + "/labels/" + mode + "/image_names.txt")
        ]
    elif dataset_name == "vot":
        image_datadir = os.path.join(DATA_DIR, "VOT2014", "dataset",)
        datadir = os.path.join(os.path.dirname(__file__), "datasets", "vot")
        gt = np.load(datadir + "/labels/" + mode + "/labels.npy")
        image_paths = [
            image_datadir + "/" + line.strip() for line in open(datadir + "/labels/" + mode + "/image_names.txt")
        ]
    return {"gt": gt, "image_paths": image_paths, "dataset": dataset}
