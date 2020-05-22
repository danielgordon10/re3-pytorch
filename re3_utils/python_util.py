import os
import time
from typing import List, Optional

import imageio
import numpy as np
import tqdm


def get_time_str():
    tt = time.localtime()
    time_str = "%04d_%02d_%02d_%02d_%02d_%02d" % (tt.tm_year, tt.tm_mon, tt.tm_mday, tt.tm_hour, tt.tm_min, tt.tm_sec)
    return time_str


def images_to_video(
    images: List[np.ndarray], output_dir: str, video_name: str, fps: int = 10, quality: Optional[float] = 5, **kwargs
):
    """Calls imageio to run FFMPEG on a list of images. For more info on
        parameters, see
        https://imageio.readthedocs.io/en/stable/format_ffmpeg.html
    Args:
        images: The list of images. Images should be HxWx3 in RGB order.
        output_dir: The folder to put the video in.
        video_name: The navme for the video.
        fps: Frames per second for the video. Not all values work with FFMPEG,
            use at your own risk.
        quality: Default is 5. Uses variable bit rate. Highest quality is 10,
            lowest is 0.  Set to None to prevent variable bitrate flags to
            FFMPEG so you can manually specify them using output_params instead.
            Specifying a fixed bitrate using ‘bitrate’ disables this parameter.
    """
    assert 0 <= quality <= 10
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video_name = os.path.join(output_dir, video_name.replace(" ", "_").replace("\n", "_") + ".mp4")
    print("Writing video to", video_name)
    writer = imageio.get_writer(video_name, fps=fps, quality=quality, **kwargs)
    for im in tqdm.tqdm(images):
        writer.append_data(im)
    writer.close()
