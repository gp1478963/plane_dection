from encoder import Decoder
from embeding import InputEmbedding
from torch import nn

class TransformerEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(TransformerEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_embedding = InputEmbedding()
        self.encoder = Decoder()

    def forward(self, x):
        x = self.input_embedding(x)
        x = self.encoder(x)
        return x