import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

use_cuda = torch.cuda.is_available()


# TODO : Takes input and produces out of same dimension our reference implementation
class FusionBiLSTM(nn.Module):
    def __init__(self, hidden_size=100, num_layers=1, batch_size=1, bidirectional=True):
        super(FusionBiLSTM, self).__init__()

        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.hidden_size = hidden_size

        self.fusion_bilstm = nn.GRU(input_size=hidden_size * 3, hidden_size=hidden_size, batch_first=True,
                                    bidirectional=bidirectional)
        self.hidden = self.init_hidden()

    def forward(self, input, mask):
        lengths = torch.sum(mask, 1)

        packed = pack_padded_sequence(input, lengths, batch_first=True)
        output, self.hidden = self.fusion_bilstm(packed, self.hidden)
        output, _ = pad_packed_sequence(output, batch_first=True)

        return output

    def init_hidden(self):
        return torch.zeros(self.num_directions * self.num_layers, self.batch_size, self.hidden_size)


class FusionBiLSTM_(nn.Module):
    def __init__(self, hidden_dim, dropout_ratio):
        super(FusionBiLSTM_, self).__init__()
        self.fusion_bilstm = nn.LSTM(3 * hidden_dim, hidden_dim, 1, batch_first=True,
                                     bidirectional=True, dropout=dropout_ratio)
        self.dropout = nn.Dropout(p=dropout_ratio)

    def forward(self, seq, mask):
        lens = torch.sum(mask, 1)
        lens_sorted, lens_argsort = torch.sort(lens, 0, True)
        _, lens_argsort_argsort = torch.sort(lens_argsort, 0)
        seq_ = torch.index_select(seq, 0, lens_argsort)
        packed = pack_padded_sequence(seq_, lens_sorted, batch_first=True)
        output, _ = self.fusion_bilstm(packed)
        e, _ = pad_packed_sequence(output, batch_first=True)
        e = e.contiguous()
        e = torch.index_select(e, 0, lens_argsort_argsort)  # B x m x 2l
        e = self.dropout(e)
        return e


if __name__ == '__main__':
    hidden_size = 100

    my_fusion = FusionBiLSTM(hidden_size)
    fusion = FusionBiLSTM_(hidden_size, 0)

    seq = torch.rand((10, 3 * hidden_size))
    mask = [1] * 10

    seq = torch.tensor(seq).unsqueeze(0)
    mask = torch.tensor(mask).unsqueeze(0)

    output = fusion(seq, mask)
    my_output = my_fusion(seq, mask)

    print(output.size(), my_output.size())
