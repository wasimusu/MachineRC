"""
Train the Coattention Network for Query Answering
"""
import os

import torch
import torch.nn as nn
import torch.optim as optim

import squad
import networks as N

from config import Config

config = Config()


def accuracy(data_iterator, model):
    """ Find the accuracy of model on given set of data """
    with torch.no_grad():
        pass


def step():
    """
    One batch of training
    :return: output, loss
    """
    pass


def train(*args, **kwargs):
    """ Train the network """

    model = N.CoattentionNetwork(config.hidden_size, num_layers=2, batch_size=1, bidrectional=False)
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    # If the network has saved model, restore it
    if os.path.exists("checkpoint"):
        state = torch.load("checkpoint")
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])

    data_iterator = []  # data iterator for squad
    for epoch in config.num_epochs:
        for i, data in enumerate(data_iterator):

            # Here goes one batch of training
            q_seq, q_mask, d_seq, d_mask, target_span = data
            model.zero_grad()
            output = model(q_seq, q_mask, d_seq, d_mask, target_span)
            loss = criterion(target_span, output)
            loss.backwards()
            optimizer.step()

            # Displaying results
            if config.print_every:
                print("Epoch : {}\tIter {}\t\tloss : {}".format(epoch, i, loss))
                with torch.no_grad():
                    pass
                # Maybe you want to do random evaluations as well for sanity check

            # Saving the model
            if config.save_every:
                state = {
                    'iter': i,
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'current_loss': loss
                }
                checkpoint_name = "checkpoint-Em{}-ep{}-it-{}".format(config.embedding_dim, epoch, i)
                torch.save(state, checkpoint_name)

if __name__ == '__main__':
    train()
