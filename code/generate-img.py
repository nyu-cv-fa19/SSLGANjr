from utils import noise
import torch
import torchvision.utils
from dataloaders import get_cifar_dataloaders
from model import Generator
#import cv2

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

filepath = 'results1/G_cifar.pt'
generator = load_generator(filepath)
_, test_loader = get_cifar_dataloaders()
num_data = len(test_loader.dataset)

# store img in generated_img folder
folder = './generated_img/'
# generator tensor
for i in range(num_data):
    output = generator(noise(128,1))
    print(output.size())
    img_name = folder + str(i+1) + 'th_img.png'
    #cv2.imwrite(img_name, output)
