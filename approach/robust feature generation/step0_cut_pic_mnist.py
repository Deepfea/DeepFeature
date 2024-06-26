import math
import os

import tensorflow as tf
from skimage import io, color
import numpy as np
from tensorflow import uint8
from tqdm import trange
import cv2

class Cluster(object):
    cluster_index = 1

    def __init__(self, h, w, l=0, a=0, b=0):
        self.update(h, w, l, a, b)
        self.pixels = []
        self.no = self.cluster_index
        Cluster.cluster_index += 1

    def update(self, h, w, l, a, b):
        self.h = h
        self.w = w
        self.l = l
        self.a = a
        self.b = b

    def __str__(self):
        return "{},{}:{} {} {} ".format(self.h, self.w, self.l, self.a, self.b)

    def __repr__(self):
        return self.__str__()


class SLICProcessor(object):
    @staticmethod
    def open_image(path):
        """
        Return:
            3D array, row col [LAB]
        """
        # print(path)
        rgb = io.imread(path)
        lab_arr = color.rgb2lab(rgb)
        return lab_arr

    @staticmethod
    def save_lab_image(path, lab_arr):
        """
        Convert the array to RBG, then save the image
        :param path:
        :param lab_arr:
        :return:
        """

        rgb_arr = color.lab2rgb(lab_arr)
        # io.imsave(path, rgb_arr)
        sr_image = rgb_arr * 255.0
        sr_image = tf.cast(sr_image, tf.int32)
        sr_image = tf.maximum(sr_image, 0)
        sr_image = tf.minimum(sr_image, 255)
        sr_image = tf.cast(sr_image, tf.uint8)
        io.imsave(path, sr_image)

    def make_cluster(self, h, w):
        h = int(h)
        w = int(w)
        return Cluster(h, w, self.data[h][w][0], self.data[h][w][1], self.data[h][w][2])

    def __init__(self, filename, K, M):
        self.K = K
        self.M = M
        self.i = 0
        self.data = self.open_image(filename)
        self.image_height = self.data.shape[0]
        self.image_width = self.data.shape[1]
        self.N = self.image_height * self.image_width
        self.S = int(math.sqrt(self.N / self.K))    # 每个像素的边长
        self.clusters = []
        self.label = {}
        self.dis = np.full((self.image_height, self.image_width), np.inf)  # 初始化距离矩阵

    # 初始化每个像素块的中心
    def init_clusters(self):
        h = self.S / 2
        w = self.S / 2
        while h < self.image_height:
            while w < self.image_width:
                self.clusters.append(self.make_cluster(h, w))
                w += self.S
            w = self.S / 2
            h += self.S


    def get_gradient(self, h, w):
        if w + 1 >= self.image_width:
            w = self.image_width - 2
        if h + 1 >= self.image_height:
            h = self.image_height - 2

        gradient = self.data[h + 1][w + 1][0] - self.data[h][w][0] + \
                   self.data[h + 1][w + 1][1] - self.data[h][w][1] + \
                   self.data[h + 1][w + 1][2] - self.data[h][w][2]
        return gradient

    def move_clusters(self):
        for cluster in self.clusters:
            cluster_gradient = self.get_gradient(cluster.h, cluster.w)
            for dh in range(-1, 2):
                for dw in range(-1, 2):
                    _h = cluster.h + dh
                    _w = cluster.w + dw
                    new_gradient = self.get_gradient(_h, _w)
                    if new_gradient < cluster_gradient:
                        cluster.update(_h, _w, self.data[_h][_w][0], self.data[_h][_w][1], self.data[_h][_w][2])
                        cluster_gradient = new_gradient

    def assignment(self):
        for cluster in self.clusters:
            for h in range(cluster.h - 2 * self.S, cluster.h + 2 * self.S):
                if h < 0 or h >= self.image_height: continue
                for w in range(cluster.w - 2 * self.S, cluster.w + 2 * self.S):
                    if w < 0 or w >= self.image_width: continue
                    L, A, B = self.data[h][w]
                    Dc = math.sqrt(
                        math.pow(L - cluster.l, 2) +
                        math.pow(A - cluster.a, 2) +
                        math.pow(B - cluster.b, 2))
                    Ds = math.sqrt(
                        math.pow(h - cluster.h, 2) +
                        math.pow(w - cluster.w, 2))
                    D = math.sqrt(math.pow(Dc / self.M, 2) + math.pow(Ds / self.S, 2))
                    if D < self.dis[h][w]:
                        if (h, w) not in self.label:
                            self.label[(h, w)] = cluster
                            cluster.pixels.append((h, w))
                        else:
                            self.label[(h, w)].pixels.remove((h, w))
                            self.label[(h, w)] = cluster
                            cluster.pixels.append((h, w))
                        self.dis[h][w] = D

    def update_cluster(self):
        for cluster in self.clusters:
            sum_h = sum_w = number = 0
            for p in cluster.pixels:
                sum_h += p[0]
                sum_w += p[1]
                number += 1
            _h = int(sum_h / number)
            _w = int(sum_w / number)
            cluster.update(_h, _w, self.data[_h][_w][0], self.data[_h][_w][1], self.data[_h][_w][2])

    def save_current_image(self, name):
        image_arr = np.copy(self.data)
        for cluster in self.clusters:
            for p in cluster.pixels:
                image_arr[p[0]][p[1]][0] = cluster.l
                image_arr[p[0]][p[1]][1] = cluster.a
                image_arr[p[0]][p[1]][2] = cluster.b
            # print(len(cluster.pixels))
            image_arr[cluster.h][cluster.w][0] = 0
            image_arr[cluster.h][cluster.w][1] = 0
            image_arr[cluster.h][cluster.w][2] = 0
            # print(len(cluster.pixels))
        # print(len(self.clusters))
        self.save_lab_image(name, image_arr)

    def iterate_10times(self):
        self.init_clusters()
        self.move_clusters()
        for i in trange(10):
            self.assignment()
            self.update_cluster()
            # name = 'lenna_M{m}_K{k}_loop{loop}.png'.format(loop=i, m=self.M, k=self.K)
            # self.save_current_image(name)

    def cutPic(self, picPath, Path, i, j):
        img = cv2.imread(picPath, 1)  # 1为彩色图片
        k = 0
        print(img.shape)
        Path = Path + os.sep + str(i)
        if not os.path.exists(Path):
            os.mkdir(Path)
        for cluster in self.clusters:
            savePath = Path + os.sep + str(i) + '_' + str(j) + '_' + str(k) + '.png'
            flag = 0
            x_x = 1000
            x_y = -1
            y_x = 1000
            y_y = -1
            for p in cluster.pixels:
                if p[0] < x_x:
                    x_x = p[0]
                if p[0] > x_y:
                    x_y = p[0]
                if p[1] < y_x:
                    y_x = p[1]
                if p[1] > y_y:
                    y_y = p[1]
            y = int(y_y) - int(y_x) + 1
            x = int(x_y) - int(x_x) + 1
            print(x)
            print(y)
            imgBlack = np.zeros((x, y, 3))
            print('all:', x_x, x_y, y_x, y_y)
            print(cluster.pixels)
            for p in cluster.pixels:
                if img[p[0]][p[1]][0] != 255 or img[p[0]][p[1]][1] != 255 or img[p[0]][p[1]][2] != 255:
                    flag = flag + 1
                    print('px:', p[0])
                    print('py:', p[1])
                    imgBlack[p[0]-x_x][p[1]-y_x][0] = img[p[0]][p[1]][0]
                    imgBlack[p[0]-x_x][p[1]-y_x][1] = img[p[0]][p[1]][1]
                    imgBlack[p[0]-x_x][p[1]-y_x][2] = img[p[0]][p[1]][2]
            imgBlack = cv2.resize(imgBlack, (28, 28))
            cv2.imwrite(savePath, imgBlack)
            k = k + 1


if __name__ == '__main__':
    pass





