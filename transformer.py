from encoder import Decoder
from embeding import InputEmbedding
from torch import nn

class TransformerEncoder(nn.Module):
    def __init__(self, input_size=224, hidden_size= 786, num_layers=8, dimension=786, layer_count=8, patch_size=16):
        super(TransformerEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_embedding = InputEmbedding(image_size=input_size, patch_size=patch_size, embedding_size=dimension)
        self.encoder = Decoder(dimension, layer_count)

    def forward(self, x):
        x = self.input_embedding(x)
        x = self.encoder(x)
        return x