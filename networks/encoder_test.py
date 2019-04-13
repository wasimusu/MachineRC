import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

use_cuda = torch.cuda.is_available()


# TODO : This and other imlementation has same output dimension ingnoring sentinel
class Encoder(nn.Module):
    """
    Generate encoding for both question and context
    """

    def __init__(self, embeddings, hidden_size=300, batch_size=1, bidirectional=False, num_layers=1):
        super(Encoder, self).__init__()

        # TODO : The bottom line is testing stuff
        # self.embeddings = nn.Embedding(10, hidden_size)     # Just put here to test things out

        self.embeddings = embeddings
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.hidden_size = hidden_size

        self.encoder = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, bidirectional=bidirectional,
                              num_layers=num_layers,
                              batch_first=True)
        self.hidden = self.init_hidden()
        self.sentinel = nn.Parameter(torch.rand(hidden_size, ))

    def forward(self, input, mask):
        lengths = torch.sum(mask, 1)

        input = self.embeddings(input)  # Convert the numbers into embeddings
        packed = pack_padded_sequence(input, lengths, batch_first=True)
        output, self.hidden = self.encoder(packed, self.hidden)
        output, _ = pad_packed_sequence(output, batch_first=True)

        # TODO: A sentinel vector should be added somehow
        return output

    def init_hidden(self):
        return torch.zeros(self.num_directions * self.num_layers, self.batch_size, self.hidden_size)


class Encoder_(nn.Module):
    def __init__(self, hidden_dim, emb_matrix, dropout_ratio):
        super(Encoder_, self).__init__()
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(10, hidden_dim)
        # self.embedding = get_pretrained_embedding(emb_matrix)
        self.emb_dim = self.embedding.embedding_dim

        self.encoder = nn.LSTM(self.emb_dim, hidden_dim, 1, batch_first=True,
                               bidirectional=False, dropout=dropout_ratio)
        self.dropout_emb = nn.Dropout(p=dropout_ratio)
        self.sentinel = nn.Parameter(torch.rand(hidden_dim, ))

    def forward(self, seq, mask):
        lens = torch.sum(mask, 1)
        lens_sorted, lens_argsort = torch.sort(lens, 0, True)
        _, lens_argsort_argsort = torch.sort(lens_argsort, 0)
        seq_ = torch.index_select(seq, 0, lens_argsort)

        seq_embd = self.embedding(seq_)
        packed = pack_padded_sequence(seq_embd, lens_sorted, batch_first=True)
        output, _ = self.encoder(packed)
        e, _ = pad_packed_sequence(output, batch_first=True)
        e = e.contiguous()
        e = torch.index_select(e, 0, lens_argsort_argsort)  # B x m x 2l
        e = self.dropout_emb(e)

        b, _ = list(mask.size())
        # copy sentinel vector at the end
        sentinel_exp = self.sentinel.unsqueeze(0).expand(b, self.hidden_dim).unsqueeze(1).contiguous()  # B x 1 x l
        lens = lens.unsqueeze(1).expand(b, self.hidden_dim).unsqueeze(1)

        sentinel_zero = torch.zeros(b, 1, self.hidden_dim)
        if use_cuda:
            sentinel_zero = sentinel_zero.cuda()
        e = torch.cat([e, sentinel_zero], 1)  # B x m + 1 x l
        e = e.scatter_(1, lens, sentinel_exp)

        return e


if __name__ == '__main__':
    myenc = Encoder(0, 100)
    enc = Encoder_(100, 0, 0)

    seq = [1, 2, 3, 4, 5, 6, 7, 8, 9, 2]
    mask = [1] * len(seq)

    seq = torch.tensor(seq).unsqueeze(0)
    mask = torch.tensor(mask).unsqueeze(0)

    output2 = enc(seq, mask)
    output = myenc(seq, mask)

    print(output.size(), output2.size())