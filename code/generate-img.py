import torch
from utils import noise
from torchvision.utils import save_image
from dataloaders import get_cifar_dataloaders
from model import Generator
from PIL import Image
import numpy as np


# reload generator model
def load_generator(filepath):
    model = Generator(128,3)
    if torch.cuda.is_available():
        model = model.cuda()
    paras = torch.load(filepath)
    model.load_state_dict(paras)
    for para in model.parameters():
        para.requires_grad = False
    
    model.eval()
    return model

filepath = 'results1/G_200th.pt'
generator = load_generator(filepath)
_, test_loader = get_cifar_dataloaders()
num_data = len(test_loader.dataset)


folder = './generated-img/'
# generator tensor
for i in range(num_data):
    output = generator(noise(128,1))
    img = output[0]
    print(img.size())
    img_name = folder + str(i+1) + 'th_img.png'
    save_image(img, img_name)
