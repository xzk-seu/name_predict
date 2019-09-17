import os
import torch
from datetime import datetime
from model import RNN
import matplotlib.pyplot as plt
from data import line2tensor, n_letters, init_cate_dict
from train import output2cate


torch.manual_seed(1)    # reproducible


def list_output(output, all_categories):
    values, indices = torch.topk(output.squeeze(), 3)
    for v, i in zip(values, indices):
        v = float(v)
        i = int(i)
        print('%s: %.3f' % (all_categories[i], v))
    print('\n')


def run():
    print(datetime.now(), 'LOAD rnn\n')
    model_path = os.path.join(os.getcwd(), 'rnn2.pkl')
    category_lines, all_categories, n_categories = init_cate_dict()
    rnn = torch.load(model_path)
    while True:
        line = input('input: ')
        line_tensor = line2tensor(line)
        line_tensor = line_tensor.view(1, -1, n_letters)
        output = rnn(line_tensor.cuda())
        guess, i = output2cate(output, all_categories)
        print(guess, '\n')
        list_output(output, all_categories)


if __name__ == '__main__':
    run()

