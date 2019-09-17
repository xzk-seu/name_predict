from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
import torch
import random


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


def init_cate_dict():
    """
    # Build the category_lines dictionary, a list of names per language
    """
    category_lines = {}
    all_categories = []
    for filename in find_files('data/names/*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = read_lines(filename)
        # 名字需要去重
        category_lines[category] = set(lines)
    n_categories = len(all_categories)
    return category_lines, all_categories, n_categories


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


def get_data_set(category_lines):
    """
    从每个category里面取出70%作为train_set
    :param category_lines:
    :return:
    """
    train_set = []
    test_set = []
    for k, v in category_lines.items():
        """
        # n = int(0.7 * len(v))
        # train_x = random.sample(v, n)
        # test_x = [x for x in v if x not in train_x]
        """

        train_x = random.choices(list(v), k=1000)
        test_x = random.choices(list(v), k=100)
        for line in train_x:
            train_set.append((line, k))
        for line in test_x:
            test_set.append((line, k))

    print('train_set size: %d\t test_set size: %d\t ' % (len(train_set), len(test_set)))
    return train_set, test_set


def run():
    category_lines, all_categories, n_categories = init_cate_dict()
    c = 0
    for k, v in category_lines.items():
        c += len(v)
        print(k, len(v))
    print('cate: %d, names: %d' % (n_categories, c))
    train_set, test_set = get_data_set(category_lines)


if __name__ == '__main__':
    run()


    """
    print(find_files('data/names/*.txt'))
    print(unicode_to_ascii('Ślusàrski'))
    print(letter2tensor('J'))
    print(letter2tensor('J').size())
    print(line2tensor('Jones').size())
    """


