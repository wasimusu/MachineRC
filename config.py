"""
Contains all the knobs for the whole project
"""
import torch

class Config:
    def __init__(self):
        """Expirement configuration"""

        self.use_cuda = torch.cuda.is_available()
        self.device = ('cuda' if self.use_cuda else 'cpu')
        self.mode = "train" # TODO: Why do we want this in config?

        self.embedding_dim = 100
        self.hidden_size = self.embedding_dim
        # TODO: Why do we need these?
        self.context_len = 200
        self.question_len = 20

        # Training
        self.num_epochs = 10
        self.learning_rate = 0.01
        self.batch_size = 100
        self.l2_norm = 0.1

        # Encoder
        self.num_encoder_layers = 2
        self.encoder_bidirectional = False
        # Fusion BiLSTM
        self.num_fusion_bilstm_layers = 2
        self.fusion_dropout_rate = 0.

        # Decoder
        self.max_dec_steps = 6
        self.num_decoder_layers = 2
        self.decoder_bidirectional = False

        # Directories
        self.data_dir = "data/"
        self.vectors_cache = "data/vectors_cache"
        self.model_dir = "model/"

        # Logs
        self.print_every = 100
        self.save_every = 100

