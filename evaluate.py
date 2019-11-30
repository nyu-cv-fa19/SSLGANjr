'''
1. load model and generate images

2. use cmd 
fid.py /path/to/img1, /path/to/img2
to compute FID score.

'''
import argparse
from tqdm import tqdm
import os
import PIL.Image as Image

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.datasets as datasets
from model import Generator, Discriminator

from data import testset,test_loader
from fid import 

'''
write generated images to a folder
'''
# load generator model ---> N
N = 2
G_filename = 'discriminator_'+ str(N) + '.pth'
D_filename = 'generator_' + str(N) + '.pth'

G_model = Generator()
G_model.load_state_dict(G_filename)
D_model = Discriminator()
D_model.eval()
D_model.load_state_dict(D_filename)

# generate images
length = len(test_loader.dataset)
os.mkdir('/generated_img')

def get_noise(size):
  mu = 0
  sigma = 1
  n = Variable(torch.Tensor(np.random.normal(0,1,size)))
  return n

for i in range(length):
    generated = G_model(get_noise(128))
    filename = str(i)
    generated.save('/generated_img/' + 'filename' + '_th.jpg', 'JPEG')


'''
write test images to a folder
'''
os.mkdir('/test_img')




