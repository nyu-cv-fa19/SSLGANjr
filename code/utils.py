import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F


# generate label
def ones_target(size):
  data = Variable(torch.ones(size,1))

  if torch.cuda.is_available():
    return data.cuda()
  return data

def zeros_target(size):
  data = Variable(torch.zeros(size,1))

  if torch.cuda.is_available():
    return data.cuda()
  return data


# generate noise
def noise(size,batch_size):
  all = []
  mu = 0
  sigma = 1
  for i in range(batch_size):
    n = np.random.normal(0,1,size)
    all.append(n)
  all = Variable(torch.Tensor(all))

  if torch.cuda.is_available():
    return all.cuda()
  return all


def generate_rotate_label(size):
  target = []
  for i in range(4):
    target = target + [i]*size
  return target


# define hinge loss function for discriminator
def get_d_loss(real_logits, fake_logits):
  D_real_loss = torch.mean(F.relu( 1 - real_logits))
  D_fake_loss = torch.mean(F.relu( 1 + fake_logits))
  D_loss = D_real_loss + D_fake_loss 
  return D_loss


def get_g_loss(fake_logits):
  G_loss = - torch.mean(fake_logits)
  return G_loss
