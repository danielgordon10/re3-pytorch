import os
import random
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from training import get_datasets

from constants import CROP_PAD
from constants import CROP_SIZE

from re3_utils.util import bb_util
from re3_utils.util import im_util
from re3_utils.util import IOU

np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)


AREA_CUTOFF = 0.25


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, num_unrolls):
        self.num_unrolls = num_unrolls
        self.all_keys = set()
        self.image_paths = []
        self.datasets = []
        self.key_lookup = {}

        self.create_keys()

    def add_dataset(self, dataset_name):
        dataset_ind = len(self.image_paths)
        data = get_datasets.get_data_for_dataset(dataset_name, "train")
        gt = data["gt"]
        num_keys = 0
        for xx in range(gt.shape[0] - self.num_unrolls):
            start_line = gt[xx, :].astype(int)
            end_line = gt[xx + self.num_unrolls, :].astype(int)
            # Check that still in the same sequence.
            # Video_id should match, track_id should match, and image number should be exactly num_unrolls frames later.
            if (
                start_line[4] == end_line[4]
                and start_line[5] == end_line[5]
                and start_line[6] + self.num_unrolls == end_line[6]
            ):
                # Add the key.
                self.all_keys.add((dataset_ind, start_line[4], start_line[5], start_line[6]))
                num_keys += 1
        print("#%s keys: %d" % (dataset_name, num_keys))

        image_paths = data["image_paths"]
        # Add the array to image_paths. Note that image paths is indexed by the dataset number THEN by the image line.
        self.image_paths.append(image_paths)

        dataset_ind = len(self.datasets)
        dataset_gt = gt
        for xx in range(dataset_gt.shape[0]):
            line = dataset_gt[xx, :].astype(int)
            self.key_lookup[(dataset_ind, line[4], line[5], line[6])] = xx
        self.datasets.append(dataset_gt)

    def create_keys(self):
        self.add_dataset("imagenet_video")

    def lookup_func(self, key):
        images = None
        labels = None
        try:
            images = []
            labels = []
            ind = key[-1]
            for dd in range(self.num_unrolls):
                path = self.image_paths[key[0]][ind + dd]
                image = cv2.imread(path)[:, :, ::-1]
                images.append(image)
                new_key = list(key)
                new_key[3] += dd
                new_key = tuple(new_key)
                image_index = self.key_lookup[new_key]
                bbox_on = self.datasets[new_key[0]][image_index, :4].copy()
                labels.append(bbox_on)
        except Exception as ex:
            import traceback

            trace = traceback.format_exc()
            print(trace)
            error_file = open("error.txt", "a+")
            error_file.write("exception in lookup_func %s\n" % str(ex))
            error_file.write(str(trace))
        finally:
            return images, labels

    def get_sample(self):
        key = random.sample(self.all_keys, 1)[0]
        images, labels = self.lookup_func(key)
        return {"images": images, "labels": labels}

    def __len__(self):
        return 2 ** 62

    def __getitem__(self, idx):
        return self.get_sample()


def collate_fn(batch):
    return batch


def get_data_loader(num_unrolls, batch_size, num_threads):
    dataset = VideoDataset(num_unrolls)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_threads, shuffle=False, pin_memory=True, collate_fn=collate_fn
    )
    return data_loader


# Make sure there is a minimum intersection with the ground truth box and the visible crop.
def fix_bbox_intersection(bbox, gtBox):
    if type(bbox) == list:
        bbox = np.array(bbox)
    if type(gtBox) == list:
        gtBox = np.array(gtBox)

    bbox = bbox.copy()

    gtBoxArea = float((gtBox[3] - gtBox[1]) * (gtBox[2] - gtBox[0]))
    bboxLarge = bb_util.scale_bbox(bbox, CROP_PAD)
    while IOU.intersection(bboxLarge, gtBox) / gtBoxArea < AREA_CUTOFF:
        bbox = bbox * 0.9 + gtBox * 0.1
        bboxLarge = bb_util.scale_bbox(bbox, CROP_PAD)
    return bbox


# Randomly jitter the box for a bit of noise.
def add_noise(bbox, prev_bbox, image_width, image_height):
    num_tries = 0
    bbox_xywh_init = bb_util.xyxy_to_xywh(bbox)
    while num_tries < 10:
        bbox_xywh = bbox_xywh_init.copy()
        center_noise = np.random.laplace(0, 1.0 / 5, 2) * bbox_xywh[[2, 3]]
        size_noise = np.clip(np.random.laplace(1, 1.0 / 15, 2), 0.6, 1.4)
        bbox_xywh[[2, 3]] *= size_noise
        bbox_xywh[[0, 1]] = bbox_xywh[[0, 1]] + center_noise
        if not (
            bbox_xywh[0] < prev_bbox[0]
            or bbox_xywh[1] < prev_bbox[1]
            or bbox_xywh[0] > prev_bbox[2]
            or bbox_xywh[1] > prev_bbox[3]
            or bbox_xywh[0] < 0
            or bbox_xywh[1] < 0
            or bbox_xywh[0] > image_width
            or bbox_xywh[1] > image_height
        ):
            num_tries = 10
        else:
            num_tries += 1
    return fix_bbox_intersection(bb_util.xywh_to_xyxy(bbox_xywh), prev_bbox)


def get_next_image_crops(images, labels, dd, noisy_box, mirrored, real_motion, network_outs):
    if network_outs is not None:
        xyxy_pred = network_outs.squeeze() / 10
        output_box = bb_util.from_crop_coordinate_system(xyxy_pred, noisy_box, CROP_PAD, 1)
        bbox_prev = noisy_box
    elif dd == 0:
        bbox_prev = labels[dd]
    else:
        bbox_prev = labels[dd - 1]

    bbox_on = labels[dd]
    if dd == 0:
        noisy_box = bbox_on.copy()
    elif not real_motion and network_outs is None:
        noisy_box = add_noise(bbox_on, bbox_on, images[0].shape[1], images[0].shape[0])
    else:
        noisy_box = fix_bbox_intersection(bbox_prev, bbox_on)

    image0 = im_util.get_cropped_input(images[max(dd - 1, 0)], bbox_prev, CROP_PAD, CROP_SIZE)[0]

    image1 = im_util.get_cropped_input(images[dd], noisy_box, CROP_PAD, CROP_SIZE)[0]

    shifted_bbox = bb_util.to_crop_coordinate_system(bbox_on, noisy_box, CROP_PAD, 1)
    shifted_bbox_xywh = bb_util.xyxy_to_xywh(shifted_bbox)
    xywh_labels = shifted_bbox_xywh
    xyxy_labels = bb_util.xywh_to_xyxy(xywh_labels) * 10
    return image0, image1, xyxy_labels, noisy_box


def get_next_image_crops_mp(images, label, dd, noisy_box, mirrored, real_motion, network_outs):
    if network_outs is not None:
        xyxy_pred = network_outs.squeeze() / 10
        output_box = bb_util.from_crop_coordinate_system(xyxy_pred, noisy_box, CROP_PAD, 1)
        bbox_prev = noisy_box
    elif dd == 0:
        bbox_prev = label[1]  # labels[dd]
    else:
        bbox_prev = label[0]  # labels[dd - 1]

    bbox_on = label[1]  # labels[dd]
    if dd == 0:
        noisy_box = bbox_on.copy()
    elif not real_motion and network_outs is None:
        noisy_box = add_noise(bbox_on, bbox_on, images[0].shape[1], images[0].shape[0])
    else:
        noisy_box = fix_bbox_intersection(bbox_prev, bbox_on)

    image0 = im_util.get_cropped_input(
        # images[max(dd-1, 0)], bbox_prev, CROP_PAD, CROP_SIZE)[0]
        images[0],
        bbox_prev,
        CROP_PAD,
        CROP_SIZE,
    )[0]

    image1 = im_util.get_cropped_input(
        # images[dd], noisy_box, CROP_PAD, CROP_SIZE)[0]
        images[1],
        noisy_box,
        CROP_PAD,
        CROP_SIZE,
    )[0]

    shifted_bbox = bb_util.to_crop_coordinate_system(bbox_on, noisy_box, CROP_PAD, 1)
    shifted_bbox_xywh = bb_util.xyxy_to_xywh(shifted_bbox)
    xywh_labels = shifted_bbox_xywh
    xyxy_labels = bb_util.xywh_to_xyxy(xywh_labels) * 10
    return image0, image1, xyxy_labels, noisy_box
