import torch
import torch.nn as nn
from model import RNN
from data import line2tensor, n_letters, init_cate_dict, get_data_set
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
learning_rate = 0.0015
EPOCH = 2
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


def batch_train(rnn, train_set, all_categories):
    start = time.time()
    for it, (x, y) in enumerate(train_set):
        category_tensor = torch.tensor([all_categories.index(y)], dtype=torch.long)
        line_tensor = line2tensor(x)
        category_tensor = category_tensor.cuda()
        line_tensor = line_tensor.cuda()
        output, loss = train(rnn, category_tensor, line_tensor)

        # Print iter number, loss, name and guess
        if it % print_every == 0:
            guess, guess_i = output2cate(output, all_categories)
            correct = '✓' if guess == y else '✗ (%s)' % y
            print('%d %.2f%% (%s) %.4f %s / %s %s' % (it, it * 100 / len(train_set), time_since(start),
                                                      loss, x, guess, correct))


def model_testing(rnn, test_set, all_categories):
    count = 0
    for it, (x, y) in enumerate(test_set):
        line_tensor = line2tensor(x)
        line_tensor = line_tensor.cuda()
        line_tensor = line_tensor.view(1, -1, n_letters)
        output = rnn(line_tensor)

        guess, guess_i = output2cate(output, all_categories)
        if guess == y:
            count += 1
    print('accuracy: %.4f' % (count/len(test_set)))


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def run():
    category_lines, all_categories, n_categories = init_cate_dict()
    rnn = RNN(n_letters, n_categories)
    rnn.cuda()

    train_set, test_set = get_data_set(category_lines)
    random.shuffle(train_set)
    for e in range(EPOCH):
        batch_train(rnn, train_set, all_categories)
        model_testing(rnn, test_set, all_categories)

    model_path = os.path.join(os.getcwd(), 'rnn2.pkl')
    torch.save(rnn, model_path)  # 保存整个网络


if __name__ == '__main__':
    run()
