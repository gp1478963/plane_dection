import os.path
import cv2
import torch
import numpy as np
import csv
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

PLANE_TYPE_ARRAY=[]

class PlaneDataset(Dataset):
    def __init__(self,csv_path, root_dir, transform=None):
        super(PlaneDataset, self).__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.csv_path = csv_path
        self.images = []
        self.images_paths = []
        self.labels = []
        self.boxes = []
        self.data_set_path = os.path.join(self.root_dir, 'dataset')
        crop_dir = os.path.join(root_dir, 'crop')
        self.PLANE_LABEL = os.listdir(crop_dir)
        self.deal_csv_reader()

    def deal_csv_reader(self):
        with open(self.csv_path, 'r') as f:
            sample_list = csv.reader(f)
            for row in sample_list:
                self.images_paths.append(os.path.join(self.data_set_path, row[0]+'.jpg'))
                self.labels.append(self.PLANE_LABEL.index(row[3]))
                self.images = cv2.imread(os.path.join(self.data_set_path, row[0]+'.jpg'))
                self.boxes.append(np.array([int(row[4]), int(row[5]), int(row[6]), int(row[7])]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = self.boxes[idx]
        label = self.labels[idx]
        if self.transform:
            image, label, boxes = self.transform(image, label, boxes)
        return image_path, image, label, boxes

def build_dataset(csv_path, root_dir, transform=None):
    train_csv = os.path.join(csv_path, 'train.csv')
    test_csv = os.path.join(csv_path, 'test.csv')
    train_dataset = PlaneDataset(csv_path=train_csv,root_dir=root_dir, transform=transform)
    test_dataset = PlaneDataset(csv_path=test_csv,root_dir=root_dir, transform=transform)
    return train_dataset, test_dataset

def convert_csv_to_text(csv_list, save_to):
    with open(save_to, 'w') as f_w:
        for csv_file in csv_list:
            with open(csv_file, 'r') as f:
                for index, line in enumerate(f.readlines()):
                    if index == 0: continue
                    f_w.write(line)

def split_dataset(root_dir, split_rate=0.8, text_save_to=None):
    import glob
    csv_list = list(glob.glob(str(root_dir) + '/*.csv'))
    np.random.shuffle(csv_list)
    train_csv_list = csv_list[:int(split_rate * len(csv_list))]
    test_csv_list = csv_list[int(split_rate * len(csv_list)):]
    convert_csv_to_text(train_csv_list, text_save_to + '/train.csv')
    convert_csv_to_text(test_csv_list, text_save_to + '/test.csv')


if __name__ == '__main__':
    # split_dataset(root_dir='D:\dataset\plane\dataset', split_rate=0.9, text_save_to='./dataset')
    train_dataset, test_dataset = build_dataset(csv_path='./dataset', root_dir='D:\dataset\plane', transform=None)
    image_path, image, label, boxes = train_dataset.__getitem__(0)
    print(image.shape)

