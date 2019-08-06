import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data

from camera import RealSenseCamera
from post_process import post_process_output
from utils.data.camera_data import CameraData
from utils.dataset_processing import evaluation

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate network')
    parser.add_argument('--network', type=str, default='epoch_07_iou_0.96', help='Path to saved network to evaluate')
    parser.add_argument('--use-depth', type=int, default=1, help='Use Depth image for evaluation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1, help='Use RGB image for evaluation (1/0)')
    parser.add_argument('--n-grasps', type=int, default=1, help='Number of grasps to consider per image')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # Connect to Camera
    logging.info('Connecting to camera...')
    cam = RealSenseCamera(device_id=830112070066)
    cam.connect()
    cam_data = CameraData(include_depth=args.use_depth, include_rgb=args.use_rgb)

    # Load Network
    logging.info('Loading model...')
    net = torch.load(args.network)
    device = torch.device("cuda:0")

    logging.info('Done')

    fig = plt.figure(figsize=(10, 10))
    while True:
        image_bundle = cam.get_image_bundle()
        rgb = image_bundle['rgb']
        depth = image_bundle['aligned_depth']
        x, depth_img, rgb_img = cam_data.get_data(rgb=rgb, depth=depth)
        with torch.no_grad():
            xc = x.to(device)
            pred = net.predict(xc)

            q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'],
                                                        pred['sin'], pred['width'])

            evaluation.plot_output(fig, cam_data.get_rgb(rgb, False), np.squeeze(depth_img), q_img, ang_img,
                                   no_grasps=args.n_grasps, grasp_width_img=width_img)

