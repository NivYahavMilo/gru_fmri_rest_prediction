import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMClassifier(nn.Module):
    def __init__(self, k_input, k_hidden, k_layers, k_class, bi_lstm, return_states=False):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(k_input, k_hidden, k_layers, bidirectional=bi_lstm, batch_first=True)
        k_hidden = k_hidden * 2 if bi_lstm else k_hidden
        self.classifier = nn.Linear(k_hidden, k_class)
        self.return_states = return_states

    def forward(self, x, x_len, max_length=None):
        x = pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)  # h0, c0 initialized to zero;
        # ignore last hidden state
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=max_length)  # ignore length
        y = self.classifier(x)

        if self.return_states:
            return x, y
        else:
            return y
