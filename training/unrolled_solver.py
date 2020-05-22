import argparse
import os
import os.path
import sys
import threading
import time

import cv2
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from training import pt_dataset
from tracker.network import Re3SmallNet
from re3_utils.pytorch_util import pytorch_util_functions as pt_util
from re3_utils.util import bb_util
from re3_utils.util import drawing
from re3_utils import python_util

from constants import CROP_PAD
from constants import CROP_SIZE
from constants import GPU_ID
from constants import LOG_DIR
from constants import OUTPUT_WIDTH
from constants import OUTPUT_HEIGHT
from re3_utils.pytorch_util import tensorboard_logger


NUM_ITERATIONS = int(1e6)
REAL_MOTION_PROB = 1.0 / 8
USE_NETWORK_PROB = 0.5


def get_next_image_crops(args):
    return pt_dataset.get_next_image_crops_mp(*args)


def main(args):
    num_unrolls = args.num_unrolls
    batch_size = args.batch_size
    timing = args.timing
    debug = args.debug or args.output

    device = pt_util.setup_devices(args.device)[0]
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=4)

    # pool = mp.Pool(min(batch_size, mp.cpu_count()))

    time_str = python_util.get_time_str()
    checkpoint_path = os.path.join(LOG_DIR, "checkpoints")
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    train_logger = None
    if not debug:
        tensorboard_dir = os.path.join(LOG_DIR, "tensorboard", time_str + "_train")
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)
        train_logger = tensorboard_logger.Logger(tensorboard_dir)

    data_loader = pt_dataset.get_data_loader(num_unrolls, batch_size, args.num_threads)
    batch_iter = iter(data_loader)

    network = Re3SmallNet(device, args)
    network.setup_optimizer(1e-5)
    network.to(device)
    network.train()

    start_iter = 0
    if args.restore:
        print("Restoring")
        start_iter = pt_util.restore_from_folder(network, checkpoint_path,)
        print("Restored", start_iter)

    if debug:
        cv2.namedWindow("debug", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("debug", OUTPUT_WIDTH, OUTPUT_HEIGHT)

    try:
        time_total = 0.000001
        num_iters = 0
        iteration = start_iter
        # Run training iterations in the main thread.

        while iteration < args.max_steps:
            if train_logger is not None and iteration % 1000 == 0:
                train_logger.network_conv_summary(network, iteration)
            if iteration == 10000:
                network.update_learning_rate(1e-6)
            if (iteration - 1) % 10 == 0:
                current_time_start = time.time()

            start_solver = time.time()
            # Timers: initial data read time | data time | forward time | backward time | total time
            timers = np.zeros(5)

            try:
                image_sequences = next(batch_iter)
            except StopIteration:
                batch_iter = iter(data_loader)
                image_sequences = next(batch_iter)
            timers[0] = time.time() - start_solver

            outputs = []
            labels = []
            images = []
            noisy_boxes = [None for _ in range(len(image_sequences))]
            mirrored = np.random.random(batch_size) < 0.5
            real_motion = np.random.random(batch_size) < REAL_MOTION_PROB
            use_network_outs = np.random.random(batch_size) < USE_NETWORK_PROB
            lstm_state = None
            network_outs = [None for _ in range(len(image_sequences))]
            for dd in range(num_unrolls):
                batch_images = []
                batch_labels = []
                process_t_start = time.time()

                for ii, vals in enumerate(image_sequences):
                    image_sequence = vals["images"]
                    label_sequence = vals["labels"]
                    image0, image1, xyxy_labels, noisy_box = pt_dataset.get_next_image_crops(
                        image_sequence,
                        label_sequence,
                        dd,
                        noisy_boxes[ii],
                        mirrored[ii],
                        real_motion[ii],
                        network_outs[ii],
                    )
                    batch_images.append((image0, image1))
                    batch_labels.append(xyxy_labels)
                    noisy_boxes[ii] = noisy_box

                images.append(batch_images)
                labels.append(batch_labels)
                image_tensor = pt_util.from_numpy(batch_images)
                timers[1] += time.time() - process_t_start
                forward_t_start = time.time()
                output = network(image_tensor, lstm_state)
                outputs.append(output)
                output = pt_util.to_numpy_array(output)
                for ii in range(batch_size):
                    if use_network_outs[ii]:
                        network_outs[ii] = output[ii]
                lstm_state = network.lstm_state
                timers[2] += time.time() - forward_t_start

            backward_t_start = time.time()
            labels = pt_util.from_numpy(labels)
            network.optimizer.zero_grad()
            outputs = torch.stack(outputs)
            loss_value = network.loss(outputs, labels.to(dtype=outputs.dtype, device=network.device))
            loss_value.backward()
            network.optimizer.step()
            loss_value = loss_value.item()
            timers[3] = time.time() - backward_t_start

            end_solver = time.time()
            timers[4] = time.time() - start_solver
            time_total += end_solver - start_solver
            per_image_timers = timers / (num_unrolls * batch_size)

            if train_logger is not None and iteration % 10 == 0:
                train_logger.dict_log(
                    {
                        "losses/loss": loss_value,
                        "stats/data_read_time": timers[0],
                        "stats/data_time": timers[1],
                        "stats/forward_time": timers[2],
                        "stats/backward_time": timers[3],
                        "stats/total_time": timers[4],
                        "per_image_stats/data_read_time": per_image_timers[0],
                        "per_image_stats/data_time": per_image_timers[1],
                        "per_image_stats/forward_time": per_image_timers[2],
                        "per_image_stats/backward_time": per_image_timers[3],
                        "per_image_stats/total_time": per_image_timers[4],
                    },
                    iteration,
                )

            num_iters += 1
            iteration += 1
            if timing and (iteration - 1) % 10 == 0:
                print("Iteration:       %d" % (iteration - 1))
                print("Loss:            %.3f" % loss_value)
                print("Average Time:    %.3f" % (time_total / num_iters))
                print("Current Time:    %.3f" % (end_solver - start_solver))
                if num_iters > 20:
                    print("Current Average: %.3f" % ((time.time() - current_time_start) / 10))
                print("")

            # Save a checkpoint and remove old ones.
            if iteration % 500 == 0 or iteration == args.max_steps:
                pt_util.save(network, LOG_DIR + "/checkpoints/iteration_%07d.pt" % iteration, num_to_keep=1)

            # Every once in a while save a checkpoint that isn't ever removed except by hand.
            if iteration % 10000 == 0 or iteration == args.max_steps:
                pt_util.save(network, LOG_DIR + "/checkpoints/long_checkpoints/iteration_%07d.pt" % iteration)
            if not debug:
                if args.run_val and (num_iters == 1 or iteration % 1000 == 0):
                    # Run a validation set eval in a separate process.
                    def test_func():
                        test_iter_on = iteration
                        print("Staring test iter", test_iter_on)
                        import subprocess
                        import json

                        command = [
                            "python",
                            "test_net.py",
                            "--video-sample-rate",
                            str(10),
                            "--no-display",
                            "-v",
                            str(args.val_device),
                        ]
                        subprocess.call(command)
                        result = json.load(open("results.json", "r"))
                        train_logger.dict_log(
                            {
                                "eval/robustness": result["robustness"],
                                "eval/lost_targets": result["lostTarget"],
                                "eval/mean_iou": result["meanIou"],
                                "eval/avg_measure": (result["meanIou"] + result["robustness"]) / 2,
                            },
                            test_iter_on,
                        )
                        os.remove("results.json")
                        print("Ending test iter", test_iter_on)

                    test_thread = threading.Thread(target=test_func)
                    test_thread.daemon = True
                    test_thread.start()
            if args.output:
                # Look at some of the outputs.
                print("new batch")
                images = (
                    np.array(images)
                    .transpose((1, 0, 2, 3, 4, 5))
                    .reshape((batch_size, num_unrolls, 2, CROP_SIZE, CROP_SIZE, 3))
                )
                labels = pt_util.to_numpy_array(labels).transpose(1, 0, 2)
                outputs = pt_util.to_numpy_array(outputs).transpose(1, 0, 2)
                for bb in range(batch_size):
                    for dd in range(num_unrolls):
                        image0 = images[bb, dd, 0, ...]
                        image1 = images[bb, dd, 1, ...]

                        label = labels[bb, dd, :]
                        xyxy_label = label / 10
                        label_box = xyxy_label * CROP_PAD

                        output = outputs[bb, dd, ...]
                        xyxy_pred = output / 10
                        output_box = xyxy_pred * CROP_PAD

                        drawing.drawRect(image0, bb_util.xywh_to_xyxy(np.full((4, 1), 0.5) * CROP_SIZE), 2, [0, 255, 0])
                        drawing.drawRect(image1, xyxy_label * CROP_SIZE, 2, [0, 255, 0])
                        drawing.drawRect(image1, xyxy_pred * CROP_SIZE, 2, [255, 0, 0])

                        plots = [image0, image1]
                        subplot = drawing.subplot(
                            plots, 1, 2, outputWidth=OUTPUT_WIDTH, outputHeight=OUTPUT_HEIGHT, border=5
                        )
                        cv2.imshow("debug", subplot[:, :, ::-1])
                        cv2.waitKey(0)
    except Exception as e:
        import traceback

        traceback.print_exc()
    finally:
        # Save if error or killed by ctrl-c.
        if not debug:
            print("Saving...")
            pt_util.save(network, LOG_DIR + "/checkpoints/iteration_%07d.pt" % iteration, num_to_keep=-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training for Re3.")
    parser.add_argument("-n", "--num-unrolls", action="store", default=2, type=int)
    parser.add_argument("-b", "--batch-size", action="store", default=64, type=int)
    parser.add_argument("-v", "--device", type=str, default=str(GPU_ID), help="Device number or string")
    parser.add_argument("-r", "--restore", action="store_true", default=False)
    parser.add_argument("-d", "--debug", action="store_true", default=False)
    parser.add_argument("-t", "--timing", action="store_true", default=False)
    parser.add_argument("-o", "--output", action="store_true", default=False)
    parser.add_argument("-c", "--clear-snapshots", action="store_true", default=False, dest="clearSnapshots")
    parser.add_argument("--num-threads", action="store", default=2, type=int)
    parser.add_argument("--run-val", action="store_true", default=False)
    parser.add_argument("--val-device", type=str, default="0", help="Device number or string for val process to use.")
    parser.add_argument("-m", "--max-steps", type=int, default=NUM_ITERATIONS, help="Number of steps to run trainer.")
    args = parser.parse_args()
    main(args)
