"""
Contains all the knobs for the whole project
"""


class Config:
    def __init__(self):
        self.num_encoder_layers = 2
        self.bidirectional_encoder = True
        self.embedding_dim = 100
        self.hidden_size = self.embedding_dim
        self.batch_size = 100
        self.learning_rate = 0.01

        # directories
        self.data_dir = "data/"
        self.model_dir = "model/"
        # Add parameters as required

        # Logs
        self.print_every = 100
        self.save_every = 100

        self.num_epochs = 10
