import torch
import torch.nn as nn
from model import RNN
from data import line2tensor, n_letters, init_cate_dict
import time
import math
import random
import os


torch.manual_seed(1)    # reproducible


def output2cate(output, all_categories):
    _, pred_y = torch.max(output, dim=1)  # 在第一个维度上的最大值以及索引
    pred_y = int(pred_y)
    return all_categories[pred_y], pred_y


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
learning_rate = 0.005
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
    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []
    for it in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = random_training_example(category_lines, all_categories)
        line_tensor = line_tensor.cuda()
        output, loss = train(rnn, category_tensor, line_tensor)
        current_loss += loss

        # Print iter number, loss, name and guess
        if it % print_every == 0:
            guess, guess_i = output2cate(output, all_categories)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (it, it / n_iters * 100, time_since(start),
                                                    loss, line, guess, correct))

        # Add current loss avg to list of losses
        if it % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

    model_path = os.path.join(os.getcwd(), 'rnn1.pkl')
    torch.save(rnn, model_path)  # 保存整个网络


if __name__ == '__main__':
    run()
