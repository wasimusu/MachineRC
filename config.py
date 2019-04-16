"""
Contains all the knobs for the whole project
"""

import torch

class Config:
    def __init__(self):
        """Expirement configuration"""

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = 100
        self.hidden_size = self.embedding_dim

        # Training
        self.num_epochs = 10
        self.learning_rate = 0.01
        self.batch_size = 100

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
        self.model_dir = "model/"

        # Logs
        self.print_every = 100
        self.save_every = 100
