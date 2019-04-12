import torch
import torch.nn as nn
import torch.nn.functional as F


class HMN(nn.Module):
    def __init__(self, hidden_size):
        super(HMN, self).__init__()

        self.fc_d = nn.Linear(hidden_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size * 2, hidden_size)  # Output dim could be 1

        # TODO : These are not correct initializations probably
        self.r = torch.rand(1, hidden_size)  # an initial value for r is required
        self.H = torch.rand(1, hidden_size)  # This is not probably the valid value of H

    def forward(self, U, S, E):
        M_1 = torch.max(self.fc1(torch.cat((U, self.r), dim=1)))
        M_2 = torch.max(self.fc2(M_1))
        self.r = torch.tanh(self.fc_d(torch.cat((self.H, S, E))))
        score = torch.max(self.fc3(torch.cat((M_1, M_2), dim=1)))
        return score


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


class CoattentionNetwork(nn.Module):
    def __init__(self, hidden_size, num_layers, batch_size, bidrectional=True):
        super(CoattentionNetwork, self).__init__()

        self.batch_size = batch_size
        self.num_directions = 2 if bidrectional else 1
        self.num_layers = num_layers

        self.fusion_bilstm = nn.LSTM(hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidrectional)
        self.hidden = self.initHidden()

    def forward(self, D, Q):
        L = D.tranpose(1, 2).bmm(Q)  # Affinity matrix
        A_Q = F.softmax(L, 1)  # row-wise normalization to get attention weights each word in question
        A_D = F.softmax(L, 2)  # column-wise softmax to get attention weights for document
        C_Q = D.bmm(A_Q)  # C_Q : l * ( n + 1)
        C_D = torch.cat((Q, C_Q), dim=1).bmm(A_D)  # C_D : 2l * (m + 1)
        DC = torch.cat((D, C_D), dim=1)
        U, _ = self.fusion_bilstm(DC, self.hidden)  # U : 2l * m
        return U

    def initHidden(self):
        return torch.zeros(self.num_directions * self.num_layers, self.batch_size, self.hidden_size)


class Encoder(nn.Module):
    def __init__(self, bidirectional, num_layers, batch_size, hidden_size, embeddings):
        super(Encoder, self).__init__()

        self.embeddings = embeddings
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.hidden_size = hidden_size

        self.gru = nn.GRU(hidden_size=hidden_size, bidirectional=bidirectional, num_layers=num_layers)
        self.hidden = self.init_hidden()
        self.fc_question = nn.Linear(hidden_size, hidden_size)  # l * ( n + 1)

    def forward(self, input):
        input = self.embeddings(input)  # Convert the numbers into embeddings
        output, self.hidden = self.gru(input, self.hidden)
        # A sentinel vector should be added somehow
        output = torch.tanh(output)  # This is for questions only
        return output

    def init_hidden(self):
        return torch.zeros(self.num_directions * self.num_layers, self.batch_size, self.hidden_size)
