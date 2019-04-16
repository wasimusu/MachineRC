"""
Contains all the knobs for the whole project
"""
import torch


class Config:
    def __init__(self):
        # RNN params
        self.num_encoder_layers = 2
        self.bidirectional_encoder = True
        self.embedding_size = 50
        self.hidden_size = self.embedding_size
        self.batch_size = 16

        self.context_len = 200
        self.question_len = 20

        # optimizer params
        self.learning_rate = 0.01
        self.l2_norm = 0.1
        self.num_epochs = 10

        # directories
        self.data_dir = "data/"
        self.vectors_cache = "data/vectors_cache"
        self.model_dir = "model/"

        # Logs
        self.print_every = 100
        self.save_every = 100

        self.use_cuda = torch.cuda.is_available()
        self.device = ('cuda' if self.use_cuda else 'cpu')
        self.mode = "train"
        # self.mode = 'inference'
