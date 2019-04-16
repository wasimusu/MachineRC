"""
Contains all the knobs for the whole project
"""
import torch


class Config:
    def __init__(self):
        """Expirement configuration"""

        self.mode = "train"  # TODO: Why do we want this in config?

        # Device params
        # self.use_cuda = torch.cuda.is_available()
        # self.device = ('cuda' if self.use_cuda else 'cpu')
        self.device = 'cpu'
        self.use_cuda = False

        # Global dimension params
        self.embedding_dim = 50
        self.hidden_size = self.embedding_dim
        self.context_len = 200  # TODO: Why do we need this?
        self.question_len = 20  # TODO: Why do we need this?

        # Training params
        self.num_epochs = 10
        self.learning_rate = 0.001
        self.batch_size = 16
        self.l2_norm = 0.1

        # Encoder params
        self.num_encoder_layers = 2
        self.encoder_bidirectional = False
        # Fusion BiLSTM params
        self.num_fusion_bilstm_layers = 2
        self.fusion_dropout_rate = 0.

        # Decoder params
        self.max_dec_steps = 6
        self.num_decoder_layers = 1
        self.decoder_bidirectional = False

        # HMN params
        self.max_pool_size = 16

        # Directories
        self.data_dir = "data/"
        self.vectors_cache = "data/vectors_cache"
        self.experiments_root_dir = "experiments/"

        # Logs
        self.print_every = 100
        self.save_every = 1000
