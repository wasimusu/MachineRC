import torch
import torch.nn as nn


class HMN(nn.Module):
    def __init__(self):
        super(HMN, self).__init__()

    def forward(self, input):
        pass


class DynamicDecoder(nn.Module):
    def __init__(self):
        super(DynamicDecoder, self).__init__()

    def forward(self, input):
        pass


class CoattentionNetwork(nn.Module):
    def __init__(self):
        super(CoattentionNetwork, self).__init__()

    def forward(self, input):
        pass


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
        self.fc_question = nn.Linear(hidden_size, hidden_size)

    def forward(self, input):
        input = self.embeddings(input)  # Convert the numbers into embeddings
        output, self.hidden = self.gru(input, self.hidden)
        output = torch.tanh(output)  # This is for questions only
        return output

    def init_hidden(self):
        return torch.zeros(self.num_directions * self.num_layers, self.batch_size, self.hidden_size)
