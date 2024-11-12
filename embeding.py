import torch
from torch import nn

class PositionEmbedding(nn.Module):
    def __init__(self, embedding_size=768, patch_count = 64):
        super(PositionEmbedding, self).__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, patch_count, embedding_size), requires_grad=True) # (1, N, D)
        self.embedding_size = embedding_size

    def forward(self, x):
        return x + self.pos_embedding

class ClassEmbedding(nn.Module):
    def __init__(self, embedding_size=768):
        super(ClassEmbedding, self).__init__()
        self.embedding_size = embedding_size
        self.cls_embedding = nn.Parameter(torch.zeros(1, 1, embedding_size), requires_grad=True)  # (1, 1, D)

    def forward(self, x):
        y = self.cls_embedding.expand(x.size(0), 1, self.embedding_size)
        x = torch.cat((y, x), dim=1)
        return x

class InputEmbedding(nn.Module):
    def __init__(self, image_size=224, patch_size=16, embedding_size=768):
        super(InputEmbedding, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_count_one_side = image_size//patch_size
        self.patch_embedding = nn.Conv2d(in_channels=3, out_channels=embedding_size, kernel_size=self.patch_size, stride=self.patch_size)
        self.cls_embedding = ClassEmbedding(embedding_size=embedding_size)
        self.pos_embedding = PositionEmbedding(embedding_size=embedding_size, patch_count=self.patch_count_one_side*self.patch_count_one_side)

    def forward(self, x):
        x = self.patch_embedding(x) # x.shape->(B,N,P,P)
        x = x.flatten(2).transpose(1, 2) # x.shape->(B,N,P2)->(B,P2,N)
        x = self.pos_embedding(x)
        x = self.cls_embedding(x)
        return x

