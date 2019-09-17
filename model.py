import torch
import torch.nn as nn


torch.manual_seed(1)    # reproducible
HIDDEN_SIZE = 512
NUM_LAYERS = 1


class RNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,  # 字符集大小,每个字符用one-hot表示
            hidden_size=HIDDEN_SIZE,  # rnn hidden unit
            num_layers=NUM_LAYERS,  # number of rnn layer 多层的话，上层的隐层的输出作为下层的输入
            batch_first=True,
        )
        self.softmax = nn.Softmax(dim=1)
        self.fc = nn.Linear(HIDDEN_SIZE, output_size)

    def forward(self, x):
        rnn_output, (h_n, c_n) = self.rnn(x, None)
        x = rnn_output[:, -1, :]
        out = self.fc(x)
        return out


if __name__ == '__main__':
    rnn = RNN(3, 10)
    print(rnn)
