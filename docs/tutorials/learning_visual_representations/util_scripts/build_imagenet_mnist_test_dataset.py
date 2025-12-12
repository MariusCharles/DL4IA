import os
import yaml
import argparse
import pprint
import sys

sys.path.insert(1, 'path/to/folder')

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from datasets import ImageNetMnist, collate_views
from learning.transformations import DataTransformation
from learning.nce_loss import nce_loss
from utils import dataset2tensor

from models.alexnet import alexnet
from models.projection_head import ProjectionHead


# Background data set
with open("path/to/configs/contrastive_training_background.yaml", 'r') as file:
    cfg = yaml.safe_load(file)
    
dataset = ImageNetMnist(
    imagenet_data_folder=cfg['imagenet_data_folder'], 
    imagenet_labels_file=cfg['imagenet_labels_file'], 
    imagenet_classes=cfg['imagenet_classes'], 
    mnist_data_folder=cfg['mnist_data_folder'],
    shared_feature=cfg['shared_feature'])

transform = DataTransformation(cfg)
dataset.transform1 = transform(['random_cropping', 'resize'])
dataset.transform2 = transform(['gaussian_blur', 'normalize'])

data_loader = torch.utils.data.DataLoader(
    dataset,
    collate_fn=collate_views,
    batch_size=16
)

data, imagenet_labels, digit_labels = [], [], []
for imgs, labels in tqdm(data_loader):
    data.append(imgs['view1'])
    imagenet_labels.append(labels['view1']['imagenet_label'])
    digit_labels.append(labels['view1']['digit_label'])

    data.append(imgs['view2'])
    imagenet_labels.append(labels['view2']['imagenet_label'])
    digit_labels.append(labels['view2']['digit_label'])

data = torch.cat(data)
imagenet_labels = torch.cat(imagenet_labels)
digit_labels = torch.cat(digit_labels)

torch.save(data, os.path.join(cfg['res_dir'], 'test_data_background.pt'))
torch.save(imagenet_labels, os.path.join(cfg['res_dir'], 'test_imagenet_labels_background.pt'))
torch.save(digit_labels, os.path.join(cfg['res_dir'], 'test_digit_labels_background.pt'))

# Digit data set
with open("path/to/configs/configs/contrastive_training_digit.yaml", 'r') as file:
    cfg = yaml.safe_load(file)
    
dataset = ImageNetMnist(
    imagenet_data_folder=cfg['imagenet_data_folder'], 
    imagenet_labels_file=cfg['imagenet_labels_file'], 
    imagenet_classes=cfg['imagenet_classes'], 
    mnist_data_folder=cfg['mnist_data_folder'],
    shared_feature=cfg['shared_feature'])

transform = DataTransformation(cfg)
dataset.view_transform = transform()

data, imagenet_labels, digit_labels = [], [], []
for imgs, labels in tqdm(data_loader):
    data.append(imgs['view1'])
    imagenet_labels.append(labels['view1']['imagenet_label'])
    digit_labels.append(labels['view1']['digit_label'])

    data.append(imgs['view2'])
    imagenet_labels.append(labels['view2']['imagenet_label'])
    digit_labels.append(labels['view2']['digit_label'])

data = torch.cat(data)
imagenet_labels = torch.cat(imagenet_labels)
digit_labels = torch.cat(digit_labels)

torch.save(data, os.path.join(cfg['res_dir'], 'test_data_digit.pt'))
torch.save(imagenet_labels, os.path.join(cfg['res_dir'], 'test_imagenet_labels_digit.pt'))
torch.save(digit_labels, os.path.join(cfg['res_dir'], 'test_digit_labels_digit.pt'))