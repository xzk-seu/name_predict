import torch
import torch.nn as nn
from data import n_letters, n_categories, letter2tensor, output2cate


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, input_, hidden):
        input_ = input_.to(self.device)
        hidden = hidden.to(self.device)
        combined = torch.cat((input_, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)

# input_t = letter2tensor('A')
# hidden_t = torch.zeros(1, n_hidden)
#
# output_, next_hidden = rnn(input_t, hidden_t)
# print(output_)
#
# t = output2cate(output_)
# print(t)

