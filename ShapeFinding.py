"""
Copyright (c) 2020-2021 WistLab.Co.Ltd
This file is a part of soft compression algorithm for gray image
ShapeFinding.py - Find the frequency set
"""

import cv2
import os
import datetime
import numpy as np
import math
import PreProcess
import itertools


def BGRtoYUV(b, g, r):
    y = (r + 2 * g + b) / 4
    y = np.floor(y)
    u = r - g
    v = b - g
    return y, u, v


def YUVtoBGR(y, u, v):
    inverse_g = y - np.floor((u + v) / 4)
    inverse_r = u + inverse_g
    inverse_b = v + inverse_g
    return inverse_b, inverse_g, inverse_r


def negative_to_positive(img):
    img_new = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] > 0:
                img_new[i][j] = 2 * img[i][j]
            elif img[i][j] < 0:
                img_new[i][j] = -2 * img[i][j] - 1
    return img_new


class DetailSearch:
    def __init__(self):
        self.input_dir = 'train'
        self.output_dir = 'frequency'
        self.pixel_size = 256
        self.layer_start = 4
        self.start = datetime.datetime.now()
        self.detail_set = {}
        self.component = 'b'

    def main(self):
        img_path_total = os.listdir(self.input_dir)
        search_num = 1
        for f in img_path_total:
            img_path = os.path.join(self.input_dir, f)
            img = cv2.imread(img_path)
            img = img.astype(np.int16)
            b, g, r = cv2.split(img)
            y, u, v = BGRtoYUV(b, g, r)
            if self.component == 'y':
                img = y
            elif self.component == 'u':
                img = u
            elif self.component == 'v':
                img = v
            img = self.new_represent(img)
            self.search_shape(img)
            end = datetime.datetime.now()
            print('\rDetail layer: Number %d is searching, the number of codewords is %d, program has run %s'
                  % (search_num, len(self.detail_set), end - self.start), end='')
            search_num = search_num + 1
        for key in range(int(math.pow(2, self.layer_start))):
            key = tuple(map(tuple, np.array([[key]])))
            if key in self.detail_set.keys():
                pass
            else:
                self.detail_set[key] = 1
        output_name = os.path.join(self.output_dir,
                                   'frequency') + '_' + 'detail' + '_' + self.component + '.txt'
        with open(output_name, 'w') as f:
            f.write(str(self.detail_set))
        print('\nA part of searching has been finished, the result has been written into %s' % output_name)

    def new_represent(self, img):
        img_original = img
        img_fill = np.zeros((img_original.shape[0] + 1, img_original.shape[1] + 1))

        img_fill[0:1, :] = np.zeros((1, img_fill.shape[1]))
        img_fill[1:img_fill.shape[0], 1:img_fill.shape[1]] = img_original
        img_fill[1:img_fill.shape[0], 0:1] = img_fill[0:img_fill.shape[0] - 1, 1:2]

        img_return = np.zeros(img_fill.shape, np.int16)

        for i in range(1, img_fill.shape[0]):
            for j in range(1, img_fill.shape[1]):
                a = int(img_fill[i][j - 1])
                b = int(img_fill[i - 1][j])
                c = int(img_fill[i - 1][j - 1])
                if c >= max(a, b):
                    img_return[i][j] = min(a, b)
                elif c < min(a, b):
                    img_return[i][j] = max(a, b)
                else:
                    img_return[i][j] = a + b - c
        img_return = img_original - img_return[1:img_return.shape[0], 1:img_return.shape[1]]
        img_return = negative_to_positive(img_return)
        img_return = img_return % math.pow(2, self.layer_start)
        return img_return

    def search_shape(self, img):
        for detail in range(int(math.pow(2, self.layer_start))):
            number = np.sum(img == detail)
            detail = tuple(map(tuple, np.array([[detail]])))
            self.renew_set(detail, number)

    def renew_set(self, detail, number):
        if number != 0:
            if detail in self.detail_set.keys():
                self.detail_set[detail] = self.detail_set[detail] + number
            else:
                self.detail_set[detail] = number


class RoughSearch:
    def __init__(self):
        self.input_dir = 'train'
        self.output_dir = 'frequency'
        self.pixel_size = 256
        self.layer_start = 4
        self.batch_size = 1
        self.start = datetime.datetime.now()
        self.rough_set = {}
        self.rough_height = 2
        self.rough_width = 2
        self.component = 'b'

    def main(self):
        img_path_total = os.listdir(self.input_dir)
        search_num = 1
        for f in img_path_total:
            img_path = os.path.join(self.input_dir, f)
            img = cv2.imread(img_path)
            img = img.astype(np.int16)
            b, g, r = cv2.split(img)
            y, u, v = BGRtoYUV(b, g, r)
            if self.component == 'y':
                img = y
            elif self.component == 'u':
                img = u
            elif self.component == 'v':
                img = v
            img = self.new_represent(img)
            img = img.astype(np.int16)

            self.search_shape(img)
            end = datetime.datetime.now()
            print('\rRough Layer: Number %d is searching, the number of codewords is %d, program has run %s'
                  % (search_num, len(self.rough_set), end - self.start), end='')
            search_num = search_num + 1
        output_name = os.path.join(self.output_dir,
                                   'frequency') + '_' + 'rough' + '_' + self.component + '.txt'
        key_range = range(int(math.pow(2, self.layer_start)))
        for key in itertools.product(key_range, repeat=self.rough_height * self.rough_width):
            key = np.array(key)
            key = np.reshape(key, (self.rough_height, self.rough_width))
            key = tuple(map(tuple, key))
            if key in self.rough_set.keys():
                pass
            else:
                self.rough_set[key] = 1
        self.rough_set = dict(sorted(self.rough_set.items(), key=lambda item: item[1], reverse=True))
        with open(output_name, 'w') as f:
            f.write(str(self.rough_set))
        print('\nA part of searching has been finished, the result has been written into %s' % output_name)

    def new_represent(self, img):
        img_original = img
        img_fill = np.zeros((img_original.shape[0] + 1, img_original.shape[1] + 1))

        img_fill[0:1, :] = np.zeros((1, img_fill.shape[1]))
        img_fill[1:img_fill.shape[0], 1:img_fill.shape[1]] = img_original
        img_fill[1:img_fill.shape[0], 0:1] = img_fill[0:img_fill.shape[0] - 1, 1:2]

        img_return = np.zeros(img_fill.shape, np.int16)

        for i in range(1, img_fill.shape[0]):
            for j in range(1, img_fill.shape[1]):
                a = int(img_fill[i][j - 1])
                b = int(img_fill[i - 1][j])
                c = int(img_fill[i - 1][j - 1])
                if c >= max(a, b):
                    img_return[i][j] = min(a, b)
                elif c < min(a, b):
                    img_return[i][j] = max(a, b)
                else:
                    img_return[i][j] = a + b - c
        img_return = img_original - img_return[1:img_return.shape[0], 1:img_return.shape[1]]
        img_return = negative_to_positive(img_return)
        img_return = img_return % math.pow(2, self.layer_start)
        return img_return

    def search_shape(self, img):
        for i in range(0, len(img), self.rough_height):
            for j in range(0, len(img[0]), self.rough_width):
                if i + self.rough_height <= len(img) and j + self.rough_width <= len(img[0]):
                    value = img[i:i + self.rough_height, j:j + self.rough_width]
                    value = tuple(map(tuple, value))
                    self.renew_set(value)

    def renew_set(self, sample):
        if sample in self.rough_set.keys():
            new_value = self.rough_set[sample] + 1
            self.rough_set[sample] = new_value
        else:
            self.rough_set[sample] = 1


class ShapeSearch:

    def __init__(self):
        self.input_dir = 'train'
        self.output_dir = 'frequency'
        self.batch_size = 1
        self.shape_height = [1, 4]
        self.shape_width = [1, 4]
        self.pixel_size = 256
        self.layer_start = 4
        self.layer_end = int(math.log2(self.pixel_size)) + 1
        self.start = datetime.datetime.now()
        self.shape_set = {}
        self.degree = 0.1
        self.component = 'b'

    def main(self):
        img_path_total = os.listdir(self.input_dir)
        round_total = math.ceil(len(img_path_total) / self.batch_size)
        round_num = 0
        for i in range(round_total):
            start_num = self.batch_size * round_num
            end_num = self.batch_size * (round_num + 1)
            try:
                img_path_batch = img_path_total[start_num: end_num]
            except IndexError:
                img_path_batch = img_path_total[start_num:]

            for f in img_path_batch:
                img_path = os.path.join(self.input_dir, f)
                img = cv2.imread(img_path)
                img = img.astype(np.int16)
                b, g, r = cv2.split(img)
                y, u, v = BGRtoYUV(b, g, r)
                if self.component == 'y':
                    img = y
                elif self.component == 'u':
                    img = u
                elif self.component == 'v':
                    img = v
                img = self.new_represent(img)
                img = img.astype(np.int16)
                self.search_shape(img)

            del_num = self.shape_compress(degree2=round_num + 1, degree3=self.batch_size)
            end = datetime.datetime.now()
            print('\rShape Layer: Number %d is searching, the number of codewords is %d, delete num is %d, '
                  'program has run %s '
                  % (round_num + 1, len(self.shape_set), del_num, end - self.start), end='')
            round_num = round_num + 1
        for key in range(1, int(math.pow(2, self.layer_end - self.layer_start))):
            key = tuple(map(tuple, np.array([[key]])))
            if key in self.shape_set.keys():
                pass
            else:
                self.shape_set[key] = 1
        shape_set = dict(sorted(self.shape_set.items(), key=lambda item: item[1], reverse=True))
        output_name = os.path.join(self.output_dir,
                                   'frequency') + '_' + 'shape' + '_' + self.component + '.txt'
        with open(output_name, 'w') as f:
            f.write(str(shape_set))
        print('\nA part of searching has been finished, the result has been written into %s' % output_name)

    def new_represent(self, img):
        img_original = img
        img_fill = np.zeros((img_original.shape[0] + 1, img_original.shape[1] + 1))

        img_fill[0:1, :] = np.zeros((1, img_fill.shape[1]))
        img_fill[1:img_fill.shape[0], 1:img_fill.shape[1]] = img_original
        img_fill[1:img_fill.shape[0], 0:1] = img_fill[0:img_fill.shape[0] - 1, 1:2]

        img_return = np.zeros(img_fill.shape, np.int16)
        for i in range(1, img_fill.shape[0]):
            for j in range(1, img_fill.shape[1]):
                a = int(img_fill[i][j - 1])
                b = int(img_fill[i - 1][j])
                c = int(img_fill[i - 1][j - 1])
                if c >= max(a, b):
                    img_return[i][j] = min(a, b)
                elif c < min(a, b):
                    img_return[i][j] = max(a, b)
                else:
                    img_return[i][j] = a + b - c
        img_return = img_original - img_return[1:img_return.shape[0], 1:img_return.shape[1]]
        img_return = negative_to_positive(img_return)
        img_return = img_return // math.pow(2, self.layer_start)
        return img_return

    def search_shape(self, img):
        lo_total = np.argwhere(img != 0)
        for lo in lo_total:
            (i, j) = lo
            for u in range(self.shape_height[0], self.shape_height[1] + 1):
                for v in range(self.shape_width[0], self.shape_width[1] + 1):
                    self.get_sample(img, [i, j, u, v])

    def get_sample(self, img, index):
        if (index[0] + index[2] <= len(img)) and (index[1] + index[3] <= len(img[0])):
            sample = img[index[0]:index[0] + index[2], index[1]:index[1] + index[3]]
            sample = tuple(map(tuple, sample))
            self.renew_set(sample)

    def renew_set(self, sample):
        sample_judge = np.array(sample) == 0
        if (np.sum(sample_judge, axis=0) <= sample_judge.shape[0] / 2).all() \
                and (np.sum(sample_judge, axis=1) <= sample_judge.shape[1] / 2).all():
            if sample in self.shape_set.keys():
                new_value = self.shape_set[sample] + 1
                self.shape_set[sample] = new_value
            else:
                self.shape_set[sample] = 1

    def shape_compress(self, degree2, degree3):
        num = 0
        for key in list(self.shape_set.keys()):
            if self.shape_set[key] <= self.degree * degree2 * degree3:
                del self.shape_set[key]
                num = num + 1
        return num


if __name__ == '__main__':
    # Folder
    input_dir = 'train'
    output_dir = 'frequency'
    # Parameters
    layer_shape = 4
    batch_size = 1
    shape_degree = 0.5
    shape_height, shape_width = ([1, 4], [1, 4])
    rough_height, rough_width = (2, 1)
    PreProcess.dir_check(output_dir, empty_flag=True)
    print("*" * 150, '\n')
    component_total = ['y', 'u', 'v']
    # Do search
    for component in component_total:
        # Detail
        FS = DetailSearch()
        FS.layer_start, FS.component = layer_shape, component
        FS.main()
        print("*" * 150, '\n')
        # Rough
        SS = RoughSearch()
        SS.layer_start, SS.component, SS.rough_height, SS.rough_width = layer_shape, component, rough_height, rough_width
        SS.main()
        print("*" * 150, '\n')
        # Shape
        TS = ShapeSearch()
        TS.shape_height, TS.shape_width, TS.layer_start, TS.batch_size, TS.degree, TS.component \
            = shape_height, shape_width, layer_shape, batch_size, shape_degree, component
        TS.main()
        print("*" * 150, '\n')
