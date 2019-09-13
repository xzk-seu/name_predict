from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
import torch


all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


def find_files(path): return glob.glob(path)


def unicode_to_ascii(s):
    """
    # Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


def read_lines(file_name):
    """
    # Read a file and split into lines
    :param file_name:
    :return:
    """
    lines_list = open(file_name, encoding='utf-8').read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines_list]


"""
# Build the category_lines dictionary, a list of names per language
"""
category_lines = {}
all_categories = []
for filename in find_files('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = read_lines(filename)
    category_lines[category] = lines
n_categories = len(all_categories)


def letter2index(letter):
    # Find letter index from all_letters, e.g. "a" = 0
    return all_letters.find(letter)


def letter2tensor(letter):
    # Just for demonstration, turn a letter into a <1 x n_letters> Tensor
    tensor = torch.zeros(1, n_letters)
    tensor[0][letter2index(letter)] = 1
    return tensor


def line2tensor(line):
    # Turn a line into a <line_length x 1 x n_letters>,
    # or an array of one-hot letter vectors
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letter2index(letter)] = 1
    return tensor


def output2cate(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


if __name__ == '__main__':
    print(find_files('data/names/*.txt'))
    print(unicode_to_ascii('Ślusàrski'))
    print(letter2tensor('J'))
    print(letter2tensor('J').size())
    print(line2tensor('Jones').size())
