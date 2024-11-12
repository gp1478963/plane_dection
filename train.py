from tarfile import data_filter

from transformer import TransformerEncoder
from plane_dataset import build_dataset
from embeding import InputEmbedding
import torch
import cv2
import numpy as np

BATCH_SIZE = 1
EPOCH_COUNT = 50
LR = 1e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def transform(image, boxes):
    image = cv2.resize(image, (224, 224))
    image = image/255.0
    image = torch.tensor(image).permute(2, 0, 1).float()
    boxes = torch.tensor(boxes)
    return image, boxes

if __name__ == '__main__':

    model = TransformerEncoder()
    model.train()

    model.to(device)
    train_dataset, test_dataset = build_dataset(csv_path='./dataset', root_dir='D:\dataset\plane', transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    for epoch in range(EPOCH_COUNT):
        for i, (image_path, image, boxes) in enumerate(dataloader):
            image = image.to(device)
            boxes = boxes.to(device)
            model.zero_grad()
            output = model(image)
            print(output.size())



