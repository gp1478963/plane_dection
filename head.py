from encoder import MultilayerPerceptron
from torch import nn

class Classifier(nn.Module):
    def __init__(self, dimension=768, n_classes=4):
        super(Classifier, self).__init__()
        self.n_classes = n_classes
        self.dimension = dimension
        self.classifier = MultilayerPerceptron(feature=[dimension, dimension, n_classes+1])

    def forward(self, x):
        x = self.classifier(x)
        return x


class BoxRegress(nn.Module):
    def __init__(self, dimension=768):
        super(BoxRegress, self).__init__()
        self.dimension = dimension
        self.boxes = MultilayerPerceptron(feature=[dimension, dimension, 4])

    def forward(self, x):
        x = self.boxes(x)
        return x