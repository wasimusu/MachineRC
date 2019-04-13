import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# TODO : Takes input and produces out of same dimension our reference implementation
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


class CoattentionNetwork(nn.Module):
    def __init__(self, hidden_size, num_layers, batch_size, bidrectional=True):
        super(CoattentionNetwork, self).__init__()

        self.batch_size = batch_size
        self.num_directions = 2 if bidrectional else 1
        self.num_layers = num_layers

        self.fusion_bilstm = FusionBiLSTM(hidden_size=hidden_size)
        self.encoder = Encoder(bidirectional=bidrectional, num_layers=num_layers, batch_size=batch_size)
        self.decoder = DynamicDecoder(hidden_size=hidden_size, batch_size=batch_size, num_layers=num_layers,
                                      bidirectional=bidrectional)
        self.fc_question = nn.Linear(hidden_size, hidden_size)  # l * ( n + 1)

        self.hidden = self.initHidden()

    def forward(self, d_seq, q_seq, d_mask, target_span=None):
        D = self.encoder(d_seq)
        Q = self.encoder(q_seq)
        Q = torch.tanh(self.fc_question(Q))  # This is for questions only

        L = D.tranpose(1, 2).bmm(Q)  # Affinity matrix
        A_Q = F.softmax(L, 1)  # row-wise normalization to get attention weights each word in question
        A_D = F.softmax(L, 2)  # column-wise softmax to get attention weights for document
        C_Q = D.bmm(A_Q)  # C_Q : l * ( n + 1)
        C_D = torch.cat((Q, C_Q), dim=1).bmm(A_D)  # C_D : 2l * (m + 1)
        C_D_t = C_D.transpose(1, 2)

        bilstm_in = torch.cat((C_D_t, D), dim=2)
        # TODO : Different than the other implementation of coattention
        U, _ = self.fusion_bilstm(bilstm_in, self.hidden)  # U : 2l * m

        return loss, starts, ends

    def initHidden(self):
        return torch.zeros(self.num_directions * self.num_layers, self.batch_size, self.hidden_size)


class HMN(nn.Module):
    def __init__(self, hidden_size):
        super(HMN, self).__init__()
        self.hidden_size = hidden_size

        self.r = nn.Linear(hidden_size * 3, hidden_size)
        self.m_t_1 = nn.Linear(hidden_size * 2, hidden_size)
        self.m_t_2 = nn.Linear(hidden_size, hidden_size)
        self.final_fc = nn.Linear(hidden_size * 2, hidden_size)  # Output dim could be 1

        # TODO : These are not correct initializations probably
        self.r = torch.rand(1, hidden_size)  # an initial value for r is required
        self.H = torch.rand(1, hidden_size)  # This is not probably the valid value of H

        self.loss = nn.CrossEntropyLoss()

    def forward(self, U, S, E, target=None):
        M_1 = torch.max(self.m_t_1(torch.cat((U, self.r), dim=1)))
        M_2 = torch.max(self.m_t_2(M_1))
        self.r = torch.tanh(self.fc_r(torch.cat((self.H, S, E))))
        score = torch.max(self.fc3(torch.cat((M_1, M_2), dim=1)))

        loss = 0
        if target:
            loss = 1

        return score, loss


class DynamicDecoder(nn.Module):
    # TODO : Everything
    def __init__(self, hidden_size, batch_size, num_layers=1, bidirectional=False):
        super(DynamicDecoder, self).__init__()

        self.num_layers = num_layers
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size

        self.alpha = HMN(hidden_size)
        self.beta = HMN(hidden_size)
        self.lstm_dec = nn.LSTM(hidden_size=hidden_size * 2)

        self.hidden = self.initHidden()

    def forward(self, input):
        S = self.alpha()
        E = self.beta()

    def initHidden(self):
        return torch.zeros(self.num_directions * self.num_layers, self.batch_size, self.hidden_size)
