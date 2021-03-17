"""
Copyright (c) 2020-2021 WistLab.Co.Ltd
This file is a part of soft compression algorithm for gray image
Encode.py - Encode images by using algorithm
"""

import cv2
import os
import PreProcess
import numpy as np
import datetime
import ast
import math
from ShapeFinding import negative_to_positive, BGRtoYUV


def Golomb(m, n):
    q = math.floor(n / m) * '1' + '0'
    k = math.ceil(math.log2(m))
    c = int(math.pow(2, k)) - m
    r = n % m
    if 0 <= r < c:
        rr = format(r, 'b').zfill(k - 1)
    else:
        rr = format(r + c, 'b').zfill(k)
    value = q + rr
    return value


def Golomb_m_search(input_list, num):
    bit_need_total = []
    for m in range(1, int(math.pow(2, num))):
        bit_need = 0
        for value in input_list:
            encode = Golomb(m, value)
            bit_need = bit_need + len(encode)
        bit_need_total.append(bit_need)
    min_index = bit_need_total.index(min(bit_need_total)) + 1
    return min_index


def new_represent_shape(layer_start, img):
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
    img_return = img_return // math.pow(2, layer_start)
    return img_return


def new_represent_rough(layer_start, img):
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
    img_return = img_return % math.pow(2, layer_start)
    return img_return


def shape_encode(img_sub, height_sub, width_sub, book):
    value = {}
    show = {}
    img_flag = np.zeros((height_sub, width_sub))
    for p in list(book.keys()):
        kernel = np.array(p, np.float32)
        p_height, p_width = kernel.shape
        dst = cv2.filter2D(img_sub, -1, kernel, anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
        can_encode_location = np.argwhere(dst == np.sum(np.power(kernel, 2)))
        for cel in can_encode_location:
            if img_flag[cel[0]][cel[1]] == 1:
                continue
            if img_sub[cel[0]: cel[0] + p_height, cel[1]: cel[1] + p_width].shape != kernel.shape:
                continue
            if (img_sub[cel[0]: cel[0] + p_height, cel[1]: cel[1] + p_width] != kernel).any():
                continue
            can_encode_flag = img_flag[cel[0]: cel[0] + p_height, cel[1]: cel[1] + p_width] == 0
            if can_encode_flag.all():
                try:
                    value[(cel[0], cel[1])] = book[p]
                    show[(cel[0], cel[1])] = p
                    img_flag[cel[0]: cel[0] + p_height, cel[1]: cel[1] + p_width] = np.ones((p_height, p_width))
                except IndexError:
                    pass
    return value, show


def shape_to_binary(code_value, height_sub, width_sub):
    binary = ''
    bit_height = len(format(height_sub, 'b'))
    bit_width = len(format(width_sub, 'b'))

    locations = list(code_value.keys())
    values = list(code_value.values())

    locations_operate = locations[:]
    for i in range(len(locations_operate)):
        locations_operate[i] = locations_operate[i][0] * width + locations_operate[i][1]
    locations_rest = locations_operate[1:]
    locations_difference = []
    for i in range(len(locations_rest)):
        locations_difference.append(locations_rest[i] - locations_operate[i])

    try:
        Golomb_m = Golomb_m_search(locations_difference[:], 10)
    except ValueError:
        Golomb_m = 0
    if len(locations) != 0 and Golomb_m == 1:
        Golomb_m = 2
    if len(locations) == 0:
        Golomb_m = 1
    binary = binary + format(Golomb_m, 'b').zfill(10)
    for i in range(len(locations)):
        if i != 0:
            locations[i] = locations_difference[i - 1]
    for i in range(len(locations)):
        if i == 0:
            binary = binary + format(locations[i][0], 'b').zfill(bit_height) + \
                     format(locations[i][1], 'b').zfill(bit_width)
            binary = binary + values[i]
        else:
            location_value = Golomb(Golomb_m, locations[i])
            binary = binary + location_value
            binary = binary + values[i]
    binary = binary + Golomb(Golomb_m, 0)
    return binary, locations_difference[1:]


def rough_to_binary(img_sub, height_sub, width_sub, book, layer_start, rough_height, rough_width):
    binary = ''
    img_flag = np.zeros((height_sub, width_sub))
    for i in range(0, len(img_sub), rough_height):
        for j in range(0, len(img_sub[0]), rough_width):
            if i + rough_height <= len(img_sub) and j + rough_width <= len(img_sub[0]):
                key = img_sub[i:i + rough_height, j:j + rough_width]
                key = tuple(map(tuple, key))
                mid_value = book[key]
                binary = binary + mid_value
                img_flag[i: i + rough_height, j: j + rough_width] = np.ones((rough_height, rough_width))
    return binary, img_flag


def detail_to_binary(img_sub, flag, book, height_sub, width_sub):
    binary = ''
    for i in range(height_sub):
        for j in range(width_sub):
            if flag[i][j] == 0:
                sample = img_sub[i:i + 1, j:j + 1]
                sample = tuple(map(tuple, sample))
                value = book[sample]
                binary = binary + value
    return binary


def space_binary(img, codebook_shape, codebook_rough, codebook_detail, layer_start, space):
    height = len(img)
    width = len(img[0])

    img_shape = new_represent_shape(layer_start, img)

    [shape_value, shape_show] = shape_encode(img_shape, height, width, codebook_shape)
    shape_value = dict(sorted(shape_value.items(), key=lambda item: (item[0][0], item[0][1])))
    binary_shape_value, location = shape_to_binary(shape_value, height, width)

    img_rough = new_represent_rough(layer_start, img)
    img_detail = img_rough

    binary_rough_value, encode_flag = rough_to_binary(img_rough, height, width, codebook_rough, layer_start,
                                                      rough_height, rough_width)
    binary_detail_value = detail_to_binary(img_detail, encode_flag, codebook_detail, height, width)
    binary_value = binary_rough_value + binary_detail_value + binary_shape_value
    return binary_value


if __name__ == '__main__':

    codebook_dir = 'codebook'
    input_dir = 'test'
    output_dir = 'test_encode'
    pixel_size = 256
    layer_start = 4
    rough_height, rough_width = (2, 1)

    layer_end = int(math.log2(pixel_size)) + 1
    start = datetime.datetime.now()

    for txt in os.listdir(codebook_dir):
        with open(os.path.join(codebook_dir, txt), 'r') as f:
            codebook = f.read()
            codebook = ast.literal_eval(codebook)
        if txt == 'codebook_detail_y.txt':
            codebook_detail_y = codebook
        elif txt == 'codebook_rough_y.txt':
            codebook_rough_y = codebook
        elif txt == 'codebook_shape_y.txt':
            codebook_shape_y = codebook
        elif txt == 'codebook_detail_u.txt':
            codebook_detail_u = codebook
        elif txt == 'codebook_rough_u.txt':
            codebook_rough_u = codebook
        elif txt == 'codebook_shape_u.txt':
            codebook_shape_u = codebook
        elif txt == 'codebook_detail_v.txt':
            codebook_detail_v = codebook
        elif txt == 'codebook_rough_v.txt':
            codebook_rough_v = codebook
        elif txt == 'codebook_shape_v.txt':
            codebook_shape_v = codebook
    print('')
    encode_num = 1
    PreProcess.dir_check(output_dir, empty_flag=True)
    PreProcess.dir_check('location', empty_flag=True)
    compress_rate = []
    b_rate = []
    g_rate = []
    r_rate = []
    for f in os.listdir(input_dir):
        num = int(f[0:f.rfind('.png')])
        img_path = os.path.join(input_dir, f)
        img = cv2.imread(img_path)
        img = img.astype(np.int16)
        b, g, r = cv2.split(img)
        y, u, v = BGRtoYUV(b, g, r)

        (height, width) = b.shape
        bit_height = len(format(height, 'b'))
        bit_width = len(format(width, 'b'))
        binary = format(bit_height, 'b').zfill(4) + format(bit_width, 'b').zfill(4)
        binary = binary + format(height, 'b').zfill(bit_height) + format(width, 'b').zfill(bit_width)
        binary = binary + format(layer_start, 'b').zfill(3)
        binary = binary + format(rough_height, 'b').zfill(3) + format(rough_width, 'b').zfill(3)

        binary_b = space_binary(y, codebook_shape_y, codebook_rough_y, codebook_detail_y, layer_start, 'y')
        binary_g = space_binary(u, codebook_shape_u, codebook_rough_u, codebook_detail_u, layer_start, 'u')
        binary_r = space_binary(v, codebook_shape_v, codebook_rough_v, codebook_detail_v, layer_start, 'v')

        binary_value = binary + binary_b + binary_g + binary_r
        output_path = os.path.join(output_dir, f[0:f.rfind('.png')]) + '.tt'
        with open(output_path, 'wb') as g:
            g.write(binary_value.encode())

        original_pixel = height * width * len(format(pixel_size - 1, 'b')) * 3
        final_pixel = len(binary_value)
        end = datetime.datetime.now()
        compress_rate.append(original_pixel / final_pixel)

        b_rate.append(original_pixel / (len(binary_b) * 3))
        g_rate.append(original_pixel / (len(binary_g) * 3))
        r_rate.append(original_pixel / (len(binary_r) * 3))

        print(
            '\rSaving image %d results, it needs %d bits firstly, now needs %d by using soft compression algorithm. '
            'Program has run %s. Average compression ratio is %0.2f, minimum is %0.3f, maximum is %0.3f, variance is '
            '%0.5f' %
            (encode_num, original_pixel, final_pixel, end - start,
                np.mean(np.array(compress_rate)), min(compress_rate), max(compress_rate), np.var(compress_rate)),
            end='')
        encode_num = encode_num + 1
    # Compression ratio
    compress_rate_path = 'compression_rate' + '.txt'
    with open(compress_rate_path, 'w') as g:
        g.write(str(compress_rate))
