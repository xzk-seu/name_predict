import torch
import torch.nn as nn
from model import rnn
from data import all_categories, category_lines, line2tensor, output2cate
import time
import math
import random


def random_training_example():
    category = random.choice(all_categories)
    line = random.choice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = line2tensor(line)
    return category, line, category_tensor, line_tensor


n_iters = 100000
print_every = 5000
plot_every = 1000


# Keep track of losses for plotting
current_loss = 0
all_losses = []

criterion = nn.NLLLoss()
learning_rate = 0.005
# If you set this too high, it might explode. If too low, it might not learn


def train(category_tensor, line_tensor):
    category_tensor = category_tensor.to(rnn.device)
    line_tensor = line_tensor.to(rnn.device)
    hidden = rnn.init_hidden()

    rnn.zero_grad()

    output = None
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


start = time.time()
for it in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = random_training_example()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    # Print iter number, loss, name and guess
    if it % print_every == 0:
        guess, guess_i = output2cate(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (it, it / n_iters * 100, time_since(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if it % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0
