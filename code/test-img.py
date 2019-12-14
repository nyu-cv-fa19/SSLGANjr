import torch
from utils import noise
from torchvision.utils import save_image
from dataloaders import get_cifar_dataloaders
from model import Generator
from PIL import Image
import numpy as np

_, test_loader = get_cifar_dataloaders()
num_data = len(test_loader.dataset)

# store img in generated_img folder
folder = './test-img/'
dataiter = iter(test_loader)

j = 0
for images in dataiter:
    # range is the batch size of dataloader
    for i in range(64):
        img_name = folder + str(j+1) + 'th_img.png'
        j += 1
        save_image(images[0][i], img_name)
