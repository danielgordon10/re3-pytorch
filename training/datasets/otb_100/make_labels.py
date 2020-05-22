import glob
import re

import numpy as np

all_lines = []
im_id = 0
video_id = 0
image_names = []
for folder in sorted(glob.glob("data/*/*")):
    im_id_start = im_id
    label_files = sorted(glob.glob(folder + "/groundtruth_rect*.txt"))
    image_files = sorted(glob.glob(folder + "/img/*.jpg"))
    image_names.extend(image_files)
    for track_id, label_file in enumerate(label_files):
        lines = [map(float, re.split("[,\s]", line.strip())) for line in open(label_file)]
        if len(lines) != len(image_files):
            print("not equal", len(lines), len(image_files), folder)
        for line in lines:
            line.extend([video_id, track_id, im_id])
            im_id += 1
        all_lines.extend(lines)
        im_id = im_id_start
        video_id += 1
    im_id += len(image_files)

all_lines = np.array(all_lines)
all_lines[:, 2] += all_lines[:, 0]
all_lines[:, 3] += all_lines[:, 1]
np.save("labels/val/labels.npy", all_lines)
ff = open("labels/val/image_names.txt", "w")
ff.write("\n".join(image_names))
ff.close()
print("done")
print("num labels", all_lines.shape[0])
print("num images", len(image_names))
