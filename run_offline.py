import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
from PIL import Image

from post_process import post_process_output
from utils.data.camera_data import CameraData
from utils.visualisation.plot import plot_results, save_results
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate network')
    parser.add_argument('--network', type=str, default='output/models/190806_2139_cornell_d/epoch_22_iou_0.94', help='Path to saved network to evaluate')
    parser.add_argument('--rgb_path', type=str, default='cornell/01/pcd0105r.png', help='RGB Image path')
    parser.add_argument('--depth_path', type=str, default='cornell/01/pcd0105d.tiff', help='Depth Image path')
    parser.add_argument('--use-depth', type=int, default=1, help='Use Depth image for evaluation (0/1)')
    parser.add_argument('--use-rgb', type=int, default=1, help='Use RGB image for evaluation (1/0)')
    parser.add_argument('--n-grasps', type=int, default=1, help='Number of grasps to consider per image')
    parser.add_argument('--save', type=int, default=0, help='Save the results')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # Load image
    logging.info('Loading image...')
    pic = Image.open(args.rgb_path, 'r')
    rgb = np.array(pic)
    pic = Image.open(args.depth_path, 'r')
    depth = np.expand_dims(np.array(pic), axis=2)

    # Load Network
    logging.info('Loading model...')
    net = torch.load(args.network)
    device = torch.device("cuda:0")
    logging.info('Done')

    img_data = CameraData(include_depth=args.use_depth, include_rgb=args.use_rgb)

    x, depth_img, rgb_img = img_data.get_data(rgb=rgb, depth=depth)

    with torch.no_grad():
        xc = x.to(device)
        pred = net.predict(xc)

        q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])

        if args.save:
            save_results(
                rgb_img=img_data.get_rgb(rgb, False),
                depth_img=np.squeeze(img_data.get_depth(depth)),
                grasp_q_img=q_img,
                grasp_angle_img=ang_img,
                no_grasps=args.n_grasps,
                grasp_width_img=width_img
            )
        else:
            fig = plt.figure(figsize=(10, 10))
            plot_results(fig=fig,
                         rgb_img=img_data.get_rgb(rgb, False),
                         grasp_q_img=q_img,
                         grasp_angle_img=ang_img,
                         no_grasps=args.n_grasps,
                         grasp_width_img=width_img)
            fig.savefig('img_result.pdf')
