"""
Train the Coattention Network for Query Answering
"""
import os

import torch
import torch.nn as nn

from config import Config

config = Config()


def accuracy(data, model):
    """ Find the accuracy of model on given set of data """
    pass


def step():
    """
    One step / batch of training
    :return: output, loss
    """
    pass


def train(*args, **kwargs):
    """ Train the network """

    # If the network has saved model, restore it
    if os.path.exists("checkpoint"):
        state_dict = torch.load("checkpoint")

    for epoch in config.num_epochs:
        for iteration in range(100):

            if config.print_every:
                pass

            if config.save_every:
                # Save and restore the model
                state_dict = {
                    "epoch": epoch,
                    "iteration": iteration,
                }
                torch.save(state_dict, "checkpoint")


# Parameters of the architecture, optimizer, loss function
criterion = nn.MSELoss()

if __name__ == '__main__':
    train()
