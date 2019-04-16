import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# TODO : Takes input and produces out of same dimension our reference implementation
class Encoder(nn.Module):
    """
    Generate encoding for both question and context
    """

    def get_pretrained_embedding(self, np_emb_matrix):
        embeddings = nn.Embedding(*np_emb_matrix.shape)
        embeddings.weight = nn.Parameter(torch.from_numpy(np_emb_matrix).float())
        embeddings.weight.requires_grad = False
        return embeddings

    def __init__(self, emb_matrix,
                batch_size,
                hidden_size,
                num_layers,
                bidirectional=False):

        super(Encoder, self).__init__()

        # TODO : The bottom line is testing stuff
        # self.embeddings = nn.Embedding(10, hidden_size)     # Just put here to test things out

        self.embeddings = self.get_pretrained_embedding(emb_matrix)
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

    def forward(self, inputs, mask):
        # Get input lengths
        lens = torch.sum(mask, 1)

        # Convert the numbers into embeddings
        inputs = self.embeddings(inputs)

        packed = pack_padded_sequence(inputs, lens, batch_first=True)
        output, self.hidden = self.encoder(packed, self.hidden)
        output, _ = pad_packed_sequence(output, batch_first=True)

        # TODO: A sentinel vector should be added somehow
        return output

    def init_hidden(self):
        return torch.zeros(self.num_directions * self.num_layers, self.batch_size, self.hidden_size)


# TODO : Takes input and produces out of same dimension our reference implementation
class FusionBiLSTM(nn.Module):
    def __init__(self, dropout_rate,
                    batch_size,
                    num_layers,
                    hidden_size):

        super(FusionBiLSTM, self).__init__()

        self.num_directions = 2
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.hidden_size = hidden_size

        self.fusion_bilstm = nn.GRU(input_size=hidden_size * 3, hidden_size=hidden_size, batch_first=True,
                                    bidirectional=True)
        self.hidden = self.init_hidden()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, inputs, mask):
        lens = torch.sum(mask, 1)  # Get input lengths

        packed = pack_padded_sequence(inputs, lens, batch_first=True)
        output, self.hidden = self.fusion_bilstm(packed, self.hidden)
        output, _ = pad_packed_sequence(output, batch_first=True)

        output = self.dropout(output)
        return output

    def init_hidden(self):
        return torch.zeros(self.num_directions * self.num_layers, self.batch_size, self.hidden_size)


class DynamicDecoder(nn.Module):
    """Predicts start and end given embedding and computes loss"""

    def __init__(self, device,
                    hidden_size,
                    batch_size,
                    max_dec_steps,
                    num_layers,
                    bidirectional=False):

        super(DynamicDecoder, self).__init__()

        self.device = device
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.max_dec_steps = max_dec_steps

        self.start_hmn = HMN(hidden_size)
        self.end_hmn = HMN(hidden_size)
        self.gru = nn.GRU(input_size=hidden_size * 2,
                          hidden_size=hidden_size,
                          batch_first=True,
                          num_layers=num_layers,
                          bidirectional=bidirectional)

        # self.hidden = self.initHidden() # for GRU

    def forward(self, U, d_mask, target_span):
        batch_indices = torch.range(self.batch_size)

        # Initialize start estimate to 0
        s_i = torch.zeros(self.batch_size).long()
        # Initialize end estimate to last word in document
        e_i = torch.sum(d_mask, 1) - 1

        # Put vectors on device
        s_i.to(self.device)
        e_i.to(self.device)
        batch_indices.to(self.device)

        # Break up target for convenience
        s_target = None
        e_target = None
        if target_span is not None:
            s_target = target_span[:, 0]
            e_target = target_span[:, 1]

        h_i = None  # hidden state of GRU
        cumulative_loss = 0.
        loss = None

        # Initialize embedding at start estimate
        u_s_i = U[batch_indices, s_i:]  # batch_size x 2l

        # Iterate getting a start and end estimate every iteration
        for _ in range(self.max_dec_steps):
            # Update embedding at end estimate
            u_e_i = U[batch_indices, e_i, :]  # batch_size x 2l
            u_cat = torch.cat((u_s_i, u_e_i), 1)  # batch_size x 4l

            # Get hidden state
            h_i = self.gru(u_cat.unsqueeze(1), h_i)[1]

            # Get new start estimate and start loss
            s_i, start_loss_i = self.start_hmn(h_i, U, s_i, u_cat, s_target)

            # Update embedding at start estimate
            u_s_i = U[batch_indices, s_i:]  # batch_size x 2l

            # Get new u_cat with updated embedding at start estimate
            u_cat = torch.cat((u_s_i, u_e_i), 1)  # batch_size x 4l

            # Get new end estimate and end loss
            e_i, end_loss_i = self.end_hmn(h_i, U, e_i, u_cat, e_target)

            # Update cumulative loss if computing loss
            if target_span is not None:
                cumulative_loss += start_loss_i + end_loss_i

        # Compute loss
        if target_span is not None:
            # Loss is the mean step loss
            loss = cumulative_loss / self.max_dec_steps
        return loss, s_i, e_i

    # def initHidden(self):
    #     return torch.zeros(self.num_directions * self.num_layers, self.batch_size, self.hidden_size)


class CoattentionNetwork(nn.Module):
    def __init__(self, device,
                    hidden_size,
                    batch_size,
                    emb_matrix,
                    num_encoder_layers,
                    num_fusion_bilstm_layers,
                    num_decoder_layers,
                    max_dec_steps,
                    fusion_dropout_rate,
                    encoder_bidirectional=False,
                    decoder_bidirectional=False):

        super(CoattentionNetwork, self).__init__()

        self.batch_size = batch_size

        self.fusion_bilstm = FusionBiLSTM(dropout_rate=fusion_dropout_rate,
                        num_layers=num_fusion_bilstm_layers,
                        hidden_size=hidden_size,
                        batch_size=batch_size)

        self.encoder = Encoder(emb_matrix=emb_matrix,
                        hidden_size=hidden_size,
                        bidirectional=encoder_bidirectional,
                        num_layers=num_encoder_layers,
                        batch_size=batch_size)

        self.decoder = DynamicDecoder(device=device,
                        hidden_size=hidden_size,
                        batch_size=batch_size,
                        max_dec_steps=max_dec_steps,
                        num_layers=num_decoder_layers,
                        bidirectional=decoder_bidirectional)

        self.fc_question = nn.Linear(hidden_size, hidden_size)  # l * ( n + 1)

    def forward(self, q_seq, q_mask, d_seq, d_mask, target_span=None):
        """Feedforward, backprop, and return loss and predictions

        Variable names match the paper except for D, Q, D_t, and Q_t
        which are the transposes of the paper's D, Q, D_t and Q_t"""

        D = self.encoder(d_seq, d_mask)
        Q = self.encoder(q_seq, q_mask)  # Named Q prime in paper
        Q = torch.tanh(self.fc_question(Q))  # This is for questions only

        D_t = torch.transpose(D, 1, 2)  # Transpose each matrix in D batch
        L = torch.bmm(Q, D_t)  # Affinity matrix

        A_Q = F.softmax(L, 1)  # row-wise normalization to get attention weights each word in question
        A_D = F.softmax(L, 2)  # column-wise softmax to get attention weights for document
        C_Q = D.bmm(D_t, A_Q)  # C_Q : B x l x (n + 1)

        Q_t = torch.transpose(Q, 1, 2)  # Transpose each matrix in Q batch
        C_D = torch.cat((Q_t, C_Q), dim=1).bmm(A_D)  # C_D : 2l * (m + 1)
        C_D_t = C_D.transpose(1, 2)

        bilstm_in = torch.cat((C_D_t, D), dim=2)

        U = self.fusion_bilstm(bilstm_in, d_mask)  # U : 2l x m

        loss, index_start, index_end = self.decoder(U, d_mask, target_span)
        return loss, index_start, index_end

    # def initHidden(self):
    #     return torch.zeros(self.num_directions * self.num_layers, self.batch_size, self.hidden_size)


# TODO : seems okay (for now) but risky to test
class HMN(nn.Module):
    def __init__(self, hidden_size):
        super(HMN, self).__init__()
        self.hidden_size = hidden_size

        # The four dense / fc layers of HMN
        self.r = nn.Linear(hidden_size * 5, hidden_size)
        self.m_t_1 = nn.Linear(hidden_size * 3, hidden_size)
        self.m_t_2 = nn.Linear(hidden_size, hidden_size)
        self.final_fc = nn.Linear(hidden_size * 2, hidden_size)  # Output dim could be 1

        self.loss = nn.CrossEntropyLoss()

    def forward(self, H, U, U_CAT, target=None):
        """
        :param H: hidden state of decoder : h
        :param U: Result of bi-lstm fusion : 2h
        :param U_CAT : concatentation of U corresponding to ith start and end position
        :param target:  ground truth / expected start and end positions : 1 pair for each item in batch
        :return:
        """
        # TODO Change to cat along axis 1? - for batch
        r = torch.tanh(self.r(torch.cat((H, U_CAT))).unsqueeze(0))  # r : 1 * l

        M_1, _ = torch.max(self.m_t_1(torch.cat((U, r), dim=1)), dim=1)
        M_1 = M_1.unsqueeze(0)  # M_1 : 1 * l

        M_2, _ = torch.max(self.m_t_2(M_1), dim=1)
        M_2 = M_2.unsqueeze(0)  # M_2 : 1 * l

        score, _ = torch.max(self.fc3(torch.cat((M_1, M_2), dim=1)), dim=1)
        score = score.unsqueeze(0)
        score = F.softmax(score, dim=1)

        _, index = torch.max(score, dim=1)

        step_loss = None
        if target:
            # TODO: Fix this?
            step_loss = self.loss(index, target)

        return index, step_loss
