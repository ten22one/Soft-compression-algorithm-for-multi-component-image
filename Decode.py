"""
Copyright (c) 2020-2021 WistLab.Co.Ltd
This file is a part of soft compression algorithm for gray image
Decode.py - Decode binary stream data into the original image
"""

import cv2
import os
import PreProcess
import numpy as np
import datetime
import ast
import math
from ShapeFinding import YUVtoBGR
from Encode import Golomb


def anti_Golomb(m, binary):
    if m == 1 and binary == '0':
        value = 0
    else:
        index = binary.find('0')
        q = index
        k = math.ceil(math.log2(m))
        c = int(math.pow(2, k)) - m

        rr = int(binary[index + 1:], 2)
        rr_length = len(binary[index + 1:])
        if rr_length == k - 1:
            r = rr
        elif rr_length == k:
            r = rr - c
        value = q * m + r
    return value


def Golomb_search(input, m):
    q_index = input.find('0')
    q = q_index
    k = math.ceil(math.log2(m))
    c = int(math.pow(2, k)) - m

    try:
        rr1 = int(input[q_index + 1: q_index + 1 + k - 1], 2)
        n1 = q * m + rr1
        r1 = n1 % m
        if 0 <= r1 < c:
            value = input[0:q_index + 1 + k - 1]
            output_value = input[q_index + 1 + k - 1:]
        else:
            value = input[0:q_index + 1 + k]
            output_value = input[q_index + 1 + k:]
    except ValueError:
        value = input[0:q_index + 1 + k]
        output_value = input[q_index + 1 + k:]

    value = anti_Golomb(m, value)
    return value, output_value


def decode(tt):
    tt = tt.decode()
    bit_height = int(tt[0:4], 2)
    bit_width = int(tt[4:8], 2)
    height = int(tt[8:8 + bit_height], 2)
    width = int(tt[8 + bit_height:8 + bit_height + bit_width], 2)
    tt = tt[8 + bit_height + bit_width:]
    layer_start = int(tt[0:3], 2)
    rough_height = int(tt[3:6], 2)
    rough_width = int(tt[6:9], 2)
    tt = tt[9:]
    img_y, tt = decode_capacity(tt, height, width, decode_rough_y, decode_detail_y,
                                decode_shape_y, bit_height, bit_width, rough_height, rough_width, layer_start)
    img_u, tt = decode_capacity(tt, height, width, decode_rough_u, decode_detail_u,
                                decode_shape_u, bit_height, bit_width, rough_height, rough_width, layer_start)
    img_v, tt = decode_capacity(tt, height, width, decode_rough_v, decode_detail_v,
                                decode_shape_v, bit_height, bit_width, rough_height, rough_width, layer_start)

    img_b, img_g, img_r = YUVtoBGR(img_y, img_u, img_v)
    img_return = cv2.merge([img_b, img_g, img_r])
    return img_return


def decode_capacity(tt, height, width, decode_rough, decode_detail, decode_shape, bit_height,
                    bit_width, rough_height, rough_width, layer_start):
    img_rough = np.zeros((height, width))

    def Golomb_search2(number, m):
        q_index = tt.find('0', number)
        q = q_index

        k = math.ceil(math.log2(m))
        c = int(math.pow(2, k)) - m

        try:
            rr1 = int(tt[q_index + 1: q_index + 1 + k - 1], 2)
            n1 = q * m + rr1
            r1 = n1 % m

            if 0 <= r1 < c:
                value = tt[0 + number:q_index + 1 + k - 1]
                output_value = q_index + 1 + k - 1
            else:
                value = tt[0 + number:q_index + 1 + k]
                output_value = q_index + 1 + k
        except ValueError:
            value = tt[0 + number:q_index + 1 + k]
            output_value = q_index + 1 + k
        value = anti_Golomb(m, value)
        return value, output_value

    # Rough
    point_num = 0
    img_flag = np.zeros((height, width))
    for i in range(0, len(img_rough), rough_height):
        for j in range(0, len(img_rough[0]), rough_width):
            if i + rough_height <= len(img_rough) and j + rough_width <= len(img_rough[0]):
                num = 1
                while True:
                    encode_value = tt[point_num:point_num + num]
                    if encode_value in decode_rough.keys():
                        rough_value = decode_rough[encode_value]
                        img_rough[i: i + rough_height, j: j + rough_width] = np.array(rough_value)
                        img_flag[i: i + rough_height, j: j + rough_width] = np.ones((rough_height, rough_width))
                        break
                    num = num + 1
                point_num = point_num + num
    tt = tt[point_num:]
    # Detail
    point_num = 0
    for i in range(len(img_rough)):
        for j in range(len(img_rough[0])):
            if img_flag[i][j] == 0:
                num = 1
                while True:
                    encode_value = tt[point_num:point_num + num]
                    if encode_value in decode_detail.keys():
                        detail_value = decode_detail[encode_value]
                        break
                    num = num + 1
                point_num = point_num + num
                img_rough[i][j] = np.array(detail_value)
    tt = tt[point_num:]
    # Shape
    img_shape = np.zeros((height, width))
    Golomb_m = int(tt[0:10], 2)
    tt = tt[10:]
    if len(tt) != 0 and Golomb_m != 1:
        first_height = int(tt[0:bit_height], 2)
        first_width = int(tt[bit_height:bit_height + bit_width], 2)
        tt = tt[bit_height + bit_width:]
        value, tt = decode_search(tt, decode_shape)
        shape_height = len(value)
        shape_width = len(value[0])
        img_shape[first_height: first_height + shape_height, first_width: first_width + shape_width] = np.array(value)
        last_location = first_height * width + first_width
    point_num = 0
    while True:
        if Golomb_m == 1:
            point_num = len(Golomb(1, 0))
            break
        if point_num == len(tt) - 1:
            break
        if Golomb_m == 0:
            break
        location_relative, num2 = Golomb_search2(point_num, Golomb_m)
        point_num = num2
        if location_relative == 0:
            break
        location = last_location + location_relative
        last_location = location
        location_height = location // width
        location_width = location % width

        num = 1
        while True:
            encode_value = tt[point_num:point_num + num]
            if encode_value in decode_shape.keys():
                value = decode_shape[encode_value]
                break
            num = num + 1
        point_num = point_num + num
        shape_height = len(value)
        shape_width = len(value[0])
        img_shape[location_height: location_height + shape_height,
        location_width: location_width + shape_width] = np.array(value)
    img_restore = restore(img_shape, img_rough, layer_start)
    tt = tt[point_num:]
    return img_restore, tt


def decode_search(input_tt, book):
    for i in range(1, len(input_tt) + 1):
        encode_value = input_tt[0:i]
        if encode_value in book.keys():
            decode_value = book[encode_value]
            return decode_value, input_tt[i:]


def restore(img1, img2, layer):
    img_positive = img1 * math.pow(2, layer) + img2
    img_negative = np.zeros(img_positive.shape)
    for i in range(len(img_negative)):
        for j in range(len(img_negative[0])):
            x = img_positive[i][j]
            if x != 0:
                if x % 2 == 0:
                    img_negative[i][j] = x / 2
                elif x % 2 == 1:
                    img_negative[i][j] = - (x + 1) / 2
    img_final = anti_predicting(img_negative)
    return img_final


def anti_predicting(img):
    img_total = np.zeros((img.shape[0] + 1, img.shape[1] + 1))
    img_predict = np.zeros((img.shape[0] + 1, img.shape[1] + 1))
    img_flag = np.zeros(img_total.shape)
    img_flag[0:1, :] = 1
    for i in range(img_total.shape[0]):
        for j in range(img_total.shape[1]):
            if img_flag[i][j] == 0:
                if j == 0:
                    img_total[i][j] = img_total[i - 1][j + 1]
                    img_flag[i][j] = 1
                else:
                    img_predict[i][j] = predict(img_total[i][j - 1], img_total[i - 1][j], img_total[i - 1][j - 1])
                    img_total[i][j] = img[i - 1][j - 1] + img_predict[i][j]
    img = img_total[1:, 1:]
    return img


def predict(a, b, c):
    if c >= max(a, b):
        value = min(a, b)
    elif c < min(a, b):
        value = max(a, b)
    else:
        value = a + b - c
    return value


def fidelity(input1, input2):
    fidelity_rate = 0
    difference = input1 - input2
    for i in range(len(difference)):
        for j in range(len(difference[0])):
            fidelity_rate = fidelity_rate + pow(difference[i][j], 2)
    fidelity_rate = fidelity_rate / (len(difference) * len(difference[0]))
    fidelity_rate = pow(fidelity_rate, 0.5)
    fidelity_rate = np.mean(fidelity_rate)
    return fidelity_rate


if __name__ == '__main__':
    os.system('python ShapeFinding.py')
    os.system('python CodeProcessing.py')
    os.system('python Encode.py')
    input_dir = 'test_encode'
    output_dir = 'test_decode'
    original_img_dir = 'test'
    codebook_dir = 'codebook'
    start = datetime.datetime.now()
    print('\r')
    PreProcess.dir_check(output_dir, empty_flag=True)

    for txt in os.listdir(codebook_dir):
        with open(os.path.join(codebook_dir, txt), 'r') as f:
            codebook = f.read()
            codebook = ast.literal_eval(codebook)
            decode_book = {v: k for k, v in codebook.items()}
        if txt == 'codebook_detail_y.txt':
            decode_detail_y = decode_book
        elif txt == 'codebook_rough_y.txt':
            decode_rough_y = decode_book
        elif txt == 'codebook_shape_y.txt':
            decode_shape_y = decode_book
        elif txt == 'codebook_detail_u.txt':
            decode_detail_u = decode_book
        elif txt == 'codebook_rough_u.txt':
            decode_rough_u = decode_book
        elif txt == 'codebook_shape_u.txt':
            decode_shape_u = decode_book
        elif txt == 'codebook_detail_v.txt':
            decode_detail_v = decode_book
        elif txt == 'codebook_rough_v.txt':
            decode_rough_v = decode_book
        elif txt == 'codebook_shape_v.txt':
            decode_shape_v = decode_book

    error_rate_total = []
    num = 1
    for f in os.listdir(input_dir):
        tt_path = os.path.join(input_dir, f)
        if os.path.splitext(tt_path)[1] == '.tt':
            with open(tt_path, 'rb') as g:
                img_encode = g.read()
            img = decode(img_encode)

            img_original_path = os.path.join(original_img_dir, f[0:f.rfind('.tt')]) + '.png'
            output_path = os.path.join(output_dir, f[0:f.rfind('.tt')]) + '.png'
            cv2.imwrite(output_path, img)

            img_original = cv2.imread(img_original_path)
            error_rate = fidelity(img_original, img)
            error_rate_total.append(error_rate)

            end = datetime.datetime.now()
            print('\rSaving image %d, root mean square error is %0.2f, average root mean square error is %0.2f, '
                  'program has run %s '
                  % (num, error_rate, np.mean(error_rate_total), end - start), end='')
            num = num + 1

    # Error rate
    error_rate_path = 'error_rate' + '.txt'
    with open(error_rate_path, 'w') as g:
        g.write(str(error_rate_total))
