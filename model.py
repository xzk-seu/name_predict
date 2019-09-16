import torch
import torch.nn as nn


torch.manual_seed(1)    # reproducible


class RNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,  # 字符集大小,每个字符用one-hot表示
            hidden_size=64,  # rnn hidden unit
            num_layers=2,  # number of rnn layer 多层的话，上层的隐层的输出作为下层的输入
            batch_first=True,
        )

        self.fc = nn.Linear(64, output_size)

    def forward(self, x):
        rnn_output, (h_n, c_n) = self.rnn(x, None)

        out = self.fc(rnn_output[:, -1, :])
        return out


if __name__ == '__main__':
    rnn = RNN(3, 10)
    print(rnn)
