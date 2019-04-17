"""
VanillaQA is highly random and thus difficult to tame or train.
"""

import torch
import torch.optim as optim
import torch.nn as nn

import os
from networks import Encoder


class Decoder(nn.Module):
    """
    Takes as input concatenation of question encoder and context encoder and outputs start point and end point
    """

    def __init__(self, embedding, output_size=2):
        super(Decoder, self).__init__()
        self.output_size = output_size

        self.positions = nn.Linear(config.hidden_size * 2, output_size)
        self.encoder = Encoder(embedding, config.batch_size, config.hidden_size, config.num_encoder_layers,
                               config.encoder_bidirectional)
        self.softmax = nn.Softmax(dim=1)
        self.qn_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.criterion = nn.MSELoss()

    def forward(self, q_seq, q_mask, d_seq, d_mask, target_span):
        Q = self.encoder(q_seq, q_mask)[:, -1, :]
        C = self.encoder(d_seq, d_mask)[:, -1, :]
        Q = torch.tanh(self.qn_linear(Q))
        U = torch.cat((Q, C), dim=1)
        positions = self.positions(U)
        starts = positions[:, 0]
        ends = positions[:, 0]
        ends += starts
        loss = self.criterion(positions, target_span.float())
        return loss, starts, ends


from data_util.vocab import get_glove
from data_util.data_batcher import get_batch_generator
from data_util.evaluate import exact_match_score, f1_score
from preprocessing.download_wordvecs import maybe_download
from config import Config
from data_utils import get_data

config = Config()

# Embeddings and word2id and id2word
glove_path = os.path.join(config.vectors_cache, "glove.6B.{}d.txt".format(config.embedding_dim))
if not os.path.exists(glove_path):
    print("\nDownloading wordvecs to {}".format(config.vectors_cache))
    if not os.path.exists(config.vectors_cache):
        os.makedirs(config.vectors_cache)
    maybe_download(config.glove_base_url, config.glove_filename, config.vectors_cache, 862182613)

emb_matrix, word2index, index2word = get_glove(glove_path, config.embedding_dim)

train_context_path = os.path.join(config.data_dir, "train.context")
train_qn_path = os.path.join(config.data_dir, "train.question")
train_ans_path = os.path.join(config.data_dir, "train.span")
dev_context_path = os.path.join(config.data_dir, "dev.context")
dev_qn_path = os.path.join(config.data_dir, "dev.question")
dev_ans_path = os.path.join(config.data_dir, "dev.span")


def step(model, optimizer, batch):
    """
    One batch of training
    :return: loss
    """
    # Here goes one batch of training
    q_seq, q_mask, d_seq, d_mask, target_span = get_data(batch, config.mode.lower() == 'train')
    model.zero_grad()
    # The loss is individual loss for each pair of question, context and answer
    loss, _, _ = model(q_seq, q_mask, d_seq, d_mask, target_span)
    loss.backward(retain_graph=True)
    optimizer.step()
    return loss


def evaluate(model, batch):
    """
    Evaluate the training and test set accuracy
    :return:
    """
    # Here goes one batch of training
    q_seq, q_mask, d_seq, d_mask, target_span = get_data(batch, config.mode.lower() == 'train')
    with torch.no_grad():
        # The loss is individual loss for each pair of question, context and answer
        loss, start_pos_pred, end_pos_pred = model(q_seq, q_mask, d_seq, d_mask, target_span)
        f1 = 0
        for i, (pred_ans_start, pred_ans_end, true_ans_tokens) in enumerate(
                zip(start_pos_pred, end_pos_pred, batch.ans_tokens)):
            pred_ans_tokens = batch.context_tokens[i][int(pred_ans_start): int(pred_ans_end) + 1]
            prediction = " ".join(pred_ans_tokens)
            ground_truth = " ".join(true_ans_tokens)
            f1 += f1_score(prediction, ground_truth)
        f1 = f1 / (i + 1)
        return f1


def train(context_path, qn_path, ans_path):
    """ Train the network """

    model = Decoder(emb_matrix, 2)
    # Select the parameters which require grad / backpropagation
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = optim.SGD(params, lr=config.learning_rate, weight_decay=config.l2_norm)

    checkpoint_name = "checkpoint-Embed{}-ep{}-iter{}".format(config.embedding_dim, 2, 1000)
    checkpoint_name = os.path.join(config.experiments_root_dir, checkpoint_name)
    # If the network has saved model, restore it
    if os.path.exists(checkpoint_name):
        state = torch.load(checkpoint_name)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start_epoch = state['epoch']
        i = state['iter']
        current_loss = state['loss']
        print("Model restored from ", checkpoint_name)
        print("Epoch : {}\tIter {}\t\tloss : {}".format(start_epoch, i, current_loss))
    else:
        print("Training with fresh parameters")

    # For each epoch
    for epoch in range(config.num_epochs):
        # For each batch
        for i, batch in enumerate(get_batch_generator(word2index, context_path, qn_path, ans_path,
                                                      config.batch_size, config.context_len,
                                                      config.question_len, discard_long=True)):

            # Take step in training
            loss = step(model, optimizer, batch)

            # Displaying results
            if i % config.print_every == 0:
                f1 = evaluate(model, batch)
                print("Epoch : {}\tIter {}\t\tloss : {}\tf1 : {}".format(epoch, i, "%.2f" % loss, "%.2f" % f1))
                # Maybe you want to do random evaluations as well for sanity check

            # Saving the model
            if i % config.save_every == 0:
                state = {
                    'iter': i,
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'current_loss': loss
                }
                checkpoint_name = "checkpoint-Embed{}-ep{}-iter{}".format(config.embedding_dim, epoch, i)
                checkpoint_name = os.path.join(config.experiments_root_dir, checkpoint_name)
                torch.save(state, checkpoint_name)


if __name__ == '__main__':
    if config.mode == 'train':
        context_path = train_context_path
        qn_path = train_qn_path
        ans_path = train_ans_path
    else:
        context_path = dev_context_path
        qn_path = dev_qn_path
        ans_path = dev_ans_path
    train(context_path, qn_path, ans_path)
