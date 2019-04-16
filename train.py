"""
Train the Coattention Network for Query Answering
"""
import os

import torch
import torch.optim as optim

import squad
import networks as N
from data_util.vocab import get_glove
from data_util.data_batcher import get_batch_generator
from data_util.evaluate import exact_match_score, f1_score
from config import Config
from data_utils import get_data

config = Config()

# Embeddings and word2id and id2word
glove_path = os.path.join(config.vectors_cache, "glove.6B.{}d.txt".format(config.embedding_size))
embeddings, word2index, index2word = get_glove(glove_path, config.embedding_size)

train_context_path = os.path.join(config.data_dir, "train.context")
train_qn_path = os.path.join(config.data_dir, "train.question")
train_ans_path = os.path.join(config.data_dir, "train.span")
dev_context_path = os.path.join(config.data_dir, "dev.context")
dev_qn_path = os.path.join(config.data_dir, "dev.question")
dev_ans_path = os.path.join(config.data_dir, "dev.span")


def step(model, optimizer, batch):
    """
    One batch of training
    :return: output, loss
    """
    # Here goes one batch of training
    q_seq, q_mask, d_seq, d_mask, target_span = get_data(batch, config.mode.lower() == 'train')
    print("q_seq : ", q_seq)
    print("q_mask : ", q_mask)
    print("d_seq : ", d_seq)
    print("target span : ", target_span)

    model.zero_grad()
    loss, _, _ = model(q_seq, q_mask, d_seq, d_mask, target_span)
    loss.backwards()
    optimizer.step()
    return loss


def train(*args, **kwargs):
    """ Train the network """

    model = N.CoattentionNetwork(config.device, config.hidden_size, config.num_encoder_layers, config.batch_size,
                                 embeddings, bidrectional=False)
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)

    # If the network has saved model, restore it
    if os.path.exists("checkpoint"):
        state = torch.load("checkpoint")
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])

    for epoch in range(config.num_epochs):
        for batch in get_batch_generator(word2index, train_context_path,
                                         train_qn_path, train_ans_path,
                                         config.batch_size, config.context_len,
                                         config.question_len, discard_long=True):

            loss = step(model, optimizer, batch)

            # Displaying results
            if config.print_every:
                print("Epoch : {}\tIter {}\t\tloss : {}".format(epoch, 1, loss))
                with torch.no_grad():
                    pass
                # Maybe you want to do random evaluations as well for sanity check

            # Saving the model
            if config.save_every:
                state = {
                    'iter': 1,
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'current_loss': loss
                }
                checkpoint_name = "checkpoint-Em{}-ep{}-it-{}".format(config.embedding_size, epoch, 1)
                torch.save(state, checkpoint_name)


if __name__ == '__main__':
    train()
