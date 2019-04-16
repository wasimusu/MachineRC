import torch
import torch.optim as optim
import torch.nn as nn

from squad import Squad


class ContextEncoder(nn.Module):
    """
    Determine representation / encoding of Context
    """

    def __init__(self, embedding_size, hidden_dim, bidirectional=False, num_layers=1, batch_size=1, vocab_size=1000):
        super(ContextEncoder, self).__init__()

        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)

        self.lstm = nn.LSTM(embedding_size, hidden_dim)
        self.softmax = nn.Softmax(dim=1)
        self.hidden = self._init_hidden()

    def forward(self, x):
        x = self.embedding(x)
        x, self.hidden = self.lstm(x, self.hidden)
        x = self.softmax(x)
        return x

    def _init_hidden(self):
        return (torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_dim),
                )


class QuestionEncoder(nn.Module):
    """
    Determine representation of Question
    """

    def __init__(self, embedding_size, hidden_dim, bidirectional=False, num_layers=1, batch_size=1, vocab_size=1000):
        super(QuestionEncoder, self).__init__()

        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)

        self.lstm = nn.LSTM(embedding_size, hidden_dim)
        self.softmax = nn.Softmax(dim=1)
        self.hidden = self._init_hidden()

    def forward(self, x):
        x = self.embedding(x)
        x, self.hidden = self.lstm(x, self.hidden)
        x = self.softmax(x)
        return x

    def _init_hidden(self):
        return (torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_dim),
                )


class Decoder(nn.Module):
    """
    Takes as input concatenation of question encoder and context encoder and outputs start point and end point
    """

    def __init__(self, feature_size, output_size=2):
        super(Decoder, self).__init__()
        self.feature_size = feature_size
        self.output_size = output_size

        self.regressor = nn.Linear(feature_size, output_size)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.nn(x)
        x = self.softmax(x)
        return x


# Architecture Details
hidden_dim = 300
embedding_size = 300
num_layers = 1
bidirectional = False

decoder_feature_size = 600  # Concat encoding of both encoders
decoder_output_size = 2

batch_size = 64
vocab_size = 10000
learning_rate = 0.001

device = ('cuda' if torch.cuda.is_available() else 'cpu')

question_encoder = QuestionEncoder(embedding_size, hidden_dim, bidirectional, num_layers, batch_size, vocab_size)
context_encoder = ContextEncoder(embedding_size, hidden_dim, bidirectional, num_layers, batch_size, vocab_size)
decoder = Decoder(decoder_feature_size, decoder_output_size)

criterion = nn.MSELoss()
question_optim = torch.optim.SGD(question_encoder.parameters(), learning_rate)
context_optim = torch.optim.SGD(context_encoder.parameters(), learning_rate)
decoder_optim = torch.optim.SGD(decoder.parameters(), learning_rate)

train_iterator = Squad(train=True, batch_size=1)
test_iterator = Squad(train=False, batch_size=1)
