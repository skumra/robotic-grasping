import socket
import time

import matplotlib.pyplot as plt
import numpy as np


class Camera(object):

    def __init__(self):
        self.im_height = 720
        self.im_width = 1280
        self.tcp_host_ip = '127.0.0.1'
        self.tcp_port = 50000
        self.buffer_size = 4098

        # Connect to server
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))

        self.intrinsics = None
        self.get_data()

    def get_data(self):

        # Ping the server with anything
        self.tcp_socket.send(b'abcd')

        # Fetch TCP data:
        #     color camera intrinsics, 9 floats, number of bytes: 9 x 4
        #     depth scale for converting depth from uint16 to float, 1 float, number of bytes: 4
        #     depth image, self.im_width x self.im_height uint16, number of bytes: self.im_width x self.im_height x 2
        #     color image, self.im_width x self.im_height x 3 uint8, number of bytes: self.im_width x self.im_height x 3
        data = b''
        while len(data) < (10*4 + self.im_height*self.im_width*5):
            data += self.tcp_socket.recv(self.buffer_size)

        # Reorganize TCP data into color and depth frame
        self.intrinsics = np.fromstring(data[0:(9*4)], np.float32).reshape(3, 3)
        depth_scale = np.fromstring(data[(9*4):(10*4)], np.float32)[0]
        depth_img = np.fromstring(data[(10*4):((10*4)+self.im_width*self.im_height*2)], np.uint16).reshape(self.im_height, self.im_width)
        color_img = np.fromstring(data[((10*4)+self.im_width*self.im_height*2):], np.uint8).reshape(self.im_height, self.im_width, 3)
        depth_img = depth_img.astype(float) * depth_scale
        return color_img, depth_img


if __name__ == '__main__':
    camera = Camera()
    time.sleep(1)  # Give camera some time to load data

    while True:
        color_img, depth_img = camera.get_data()
        plt.subplot(211)
        plt.imshow(color_img)
        plt.subplot(212)
        plt.imshow(depth_img)
        plt.show()
