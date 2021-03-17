"""
Copyright (c) 2020-2021 WistLab.Co.Ltd
This file is a part of soft compression algorithm for gray image
CodeProcessing.py - Generate the codebook according to the frequency
"""

import os
import numpy as np
import ast
import PreProcess
import cv2


class Node:
    def __init__(self, freq):
        self.left = None
        self.right = None
        self.father = None
        self.freq = freq

    def isLeft(self):
        return self.father.left == self


def createNodes(freqs):
    return [Node(freq) for freq in freqs]


def createHuffmanTree(nodes):
    num = 1
    queue = nodes[:]
    while len(queue) > 1:
        queue.sort(key=lambda item: item.freq)
        node_left = queue.pop(0)
        node_right = queue.pop(0)
        node_father = Node(node_left.freq + node_right.freq)
        num = num + 1
        node_father.left = node_left
        node_father.right = node_right
        node_left.father = node_father
        node_right.father = node_father
        queue.append(node_father)
    queue[0].father = None
    return queue[0]


# Huffman
def huffmanEncoding(nodes, root):
    codes = [''] * len(nodes)
    for i in range(len(nodes)):
        node_tmp = nodes[i]
        while node_tmp != root:
            if node_tmp.isLeft():
                codes[i] = '0' + codes[i]
            else:
                codes[i] = '1' + codes[i]
            node_tmp = node_tmp.father
    return codes


def get_keys(d, value):
    return [k for k, v in d.items() if v == value]


def codebook_delete(book):
    book_return = dict(codebook.items())
    for key in list(book.keys()):
        key_same_total = get_keys(book, book[key])
        key = np.array(key, np.float32)
        for key_same in key_same_total:
            kernel = np.array(key_same, np.float32)
            if kernel.size == 1:
                continue
            if kernel.shape == key.shape and (kernel == key).all():
                continue
            if kernel.shape[0] > key.shape[0] or kernel.shape[1] > key.shape[1]:
                continue
            dst = cv2.filter2D(key, -1, kernel, anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
            same_location = np.argwhere(dst == np.sum(np.power(kernel, 2)))
            for sl in same_location:
                if key[sl[0]: sl[0] + kernel.shape[0], sl[1]: sl[1] + kernel.shape[1]].shape == kernel.shape \
                        and (key[sl[0]: sl[0] + kernel.shape[0], sl[1]: sl[1] + kernel.shape[1]] == kernel).all():
                    try:
                        del book_return[key_same]
                    except KeyError:
                        pass
    return book_return


if __name__ == '__main__':

    input_dir = 'frequency'
    output_dir = 'codebook'
    PreProcess.dir_check(output_dir, empty_flag=True)
    for frequency in os.listdir(input_dir):
        frequency_name = os.path.join(input_dir, frequency)
        with open(frequency_name, 'r') as f:
            codebook = f.read()
            codebook = ast.literal_eval(codebook)
        print('reading %s,' % frequency_name, 'the number of codewords is %d' % len(codebook))

        frequency_class = frequency[0:frequency.rfind('_')]
        if frequency_class == 'frequency_detail':
            codebook = dict(sorted(codebook.items(), key=lambda item: item[1], reverse=True))
        elif frequency_class == 'frequency_rough':
            codebook = dict(sorted(codebook.items(), key=lambda item: item[1], reverse=True))
        elif frequency_class == 'frequency_shape':
            initial_shape_kind = len(codebook)
            codebook = codebook_delete(codebook)
            final_shape_kind = len(codebook)
            for key in codebook.keys():
                mid_value = codebook[key] * np.sum(key != 0)
                codebook[key] = mid_value
            codebook = dict(sorted(codebook.items(), key=lambda item: np.sum(np.array(item[0]) != 0), reverse=True))
        chars = list(codebook.keys())
        freqs = list(codebook.values())
        # Huffman coding
        nodes = createNodes(freqs)
        root = createHuffmanTree(nodes)
        codes = huffmanEncoding(nodes, root)
        codebook = dict(zip(chars, codes))
        output_name = os.path.join(output_dir, 'codebook') + frequency[frequency.find('_'):]
        with open(output_name, 'w') as f:
            f.write(str(codebook))
        print('Codebook has been saved as %s' % output_name)
