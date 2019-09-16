import torch
import torch.nn as nn
from model import RNN
from data import line2tensor, n_letters, init_cate_dict
import time
import math
import random
import os
import numpy as np


torch.manual_seed(1)    # reproducible


def output2cate(output, all_categories):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


def get_batch_train_data(category_lines, all_categories, batch_size):
    categories = random.choices(all_categories, k=batch_size)
    line_tensors_list = []
    category_tensors = []
    for category in categories:
        line = random.choice(category_lines[category])
        line_tensor = line2tensor(line)
        line_tensor = line_tensor.view(1, -1, n_letters)
        line_tensors_list.append(line_tensor)
        category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
        category_tensors.append(category_tensor)
    line_tensors = torch.cat(line_tensors_list)
    line_tensors_list = torch.from_numpy(np.array(line_tensors_list))
    category_tensors = torch.stack(category_tensors, dim=0)
    return line_tensors, category_tensors


def random_training_example(category_lines, all_categories):
    """

    :param category_lines: dict
    :param all_categories:  list
    :return:
    """
    category = random.choice(all_categories)
    line = random.choice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = line2tensor(line)
    return category, line, category_tensor, line_tensor


n_iters = 10000
print_every = 50
plot_every = 1000
learning_rate = 0.1
# If you set this too high, it might explode. If too low, it might not learn


def train(rnn, category_tensor, line_tensor):
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    line_tensor = line_tensor.view(1, -1, n_letters)
    output = rnn(line_tensor)

    category_tensor = category_tensor.cuda()
    loss = criterion(output, category_tensor)
    optimizer.zero_grad()  # clear gradients for this training step
    loss.backward()  # back-propagation, compute gradients
    optimizer.step()  # apply gradients

    return output, loss.item()


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def run():
    start = time.time()
    category_lines, all_categories, n_categories = init_cate_dict()
    rnn = RNN(n_letters, n_categories)
    rnn.cuda()

    line_tensors, category_tensors = get_batch_train_data(category_lines, all_categories, 100)
    line_tensors = line_tensors.cuda()
    category_tensors = category_tensors.cuda()
    for it in range(1, n_iters + 1):
        output, loss = train(rnn, category_tensors, line_tensors)

        # Print iter number, loss, name and guess
        if it % print_every == 0:
            print('%d %d%% (%s) %.4f' % (it, it / n_iters * 100, time_since(start), loss))

    model_path = os.path.join(os.getcwd(), 'rnn1.pkl')
    torch.save(rnn, model_path)  # 保存整个网络


if __name__ == '__main__':
    run()
