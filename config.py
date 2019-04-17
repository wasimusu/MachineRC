"""
Contains all the knobs for the whole project
"""
import torch


class Config:
    def __init__(self):
        """Expirement configuration"""

        self.mode = "dev"

        # Device params
        self.use_cuda = torch.cuda.is_available()
        self.device = ('cuda' if self.use_cuda else 'cpu')
        self.device = 'cpu'
        self.use_cuda = False

        # Global dimension params
        self.embedding_dim = 50
        self.hidden_size = self.embedding_dim
        self.context_len = 600
        self.question_len = 30

        # Training params
        self.num_epochs = 10
        self.learning_rate = 0.00001
        self.batch_size = 32
        self.l2_norm = 0.1
        self.max_grad_norm = 5

        # Encoder params
        self.num_encoder_layers = 2
        self.encoder_bidirectional = False
        # Fusion BiLSTM params
        self.num_fusion_bilstm_layers = 2
        self.fusion_dropout_rate = 0.1

        # Decoder params
        self.max_dec_steps = 4
        self.num_decoder_layers = 1
        self.decoder_bidirectional = False

        # HMN params
        self.max_pool_size = 16

        # Directories
        self.data_dir = "data/"
        self.vectors_cache = "data/vectors_cache"
        self.experiments_root_dir = "experiments/"

        # Logs
        self.print_every = 5
        self.save_every = 100

        # Vectors
        self.glove_base_url = "http://nlp.stanford.edu/data/"
        self.glove_filename = "glove.6B.zip"

        # Restore or run a fresh training
        self.restore = True
