import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from model import Generator, Discriminator
from data import train_loader, dataset
import torch.nn.functional as F

# define loss
epochs = 10

# define rotation loss
weight_rotation_loss_d = 1
weight_rotation_loss_g = 0.2

# check cuda availability
if torch.cuda.is_available():
  print('running on cuda')
else:
  print('running on cpu')

# generate noise
noise_size = 128
def noise(size):
  all = []
  mu = 0
  sigma = 1
  for i in range(64):
    #n = Variable(torch.Tensor(np.random.normal(0,1,size)))
    n = np.random.normal(0,1,size)
    all.append(n)
  all = Variable(torch.Tensor(all))

  if torch.cuda.is_available():
    return all.cuda()
  return all

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

def generate_rotate_label(size):
  target = []
  for i in range(4):
    target = target + [i]*size
  return target

# define hinge loss function for discriminator
def get_d_loss(real_logits, fake_logits):
  D_real_loss = torch.mean(F.relu( 1 - real_logits))
  D_fake_loss = torch.mean(F.relu( 1 + fake_logits))
  D_loss = -D_real_loss + D_fake_loss 
  return D_loss

def get_g_loss(fake_logits):
  G_loss = - torch.mean(fake_logits)
  return G_loss
  

alpha = 0.2
beta = 1

batch_size = 64
num_examples = 64
ROTATE_NUM = 4

if dataset == 'mnist':
  in_channels = 1
else:
  in_channels = 3

# get discriminator obj and generator obj
discriminator = Discriminator(in_channels)
generator = Generator(128,in_channels,32) # in_channels: same as out_channels

if torch.cuda.is_available():
  discriminator.cuda()
  generator.cuda()

# optimizer
optimizer_G = torch.optim.Adam(discriminator.parameters(), lr = 0.0002)
optimizer_D = torch.optim.Adam(generator.parameters(), lr = 0.0002)

# training
for epoch in range(epochs):

  for nth_batch, (real_batch,_) in enumerate(train_loader):

    N = real_batch.size(0)

    # prepare real data and fake data
    real_data = Variable(real_batch)
    fake_data = generator(noise(noise_size))

    if torch.cuda.is_available():
      real_data = real_data.cuda()
      fake_data = fake_data.cuda()
    
    # ---------------------------generator training--------------------------------
    optimizer_G.zero_grad()
    #discriminator(real_data)

    # true/false loss for G
    _,pred,_,_ = discriminator(fake_data)
    #G_loss = loss(x1, zeros_target(N))
    G_loss = get_g_loss(pred)

    fake_data90 = torch.rot90(fake_data, 3, [2,3])
    fake_data180 = torch.rot90(fake_data, 2, [2,3])
    fake_data270 = torch.rot90(fake_data,1, [2,3])
    fake_data_rot = torch.cat((fake_data, fake_data90, fake_data180, fake_data270),0)

    num_examples = generate_rotate_label(64) #[0,0,0,0,1,1,1,1,2,2,2,2]
    _,_,h,_ = discriminator(fake_data_rot)
    pred_rot = torch.log(h + 1e-10)

    # generate one hot vector according to rot label
    one_hot_label = torch.zeros([64*4,ROTATE_NUM], dtype = torch.float32)
    for i in range(64*4):
      if num_examples[i] == 0:
        one_hot_label[i][0] = 1
      if num_examples[i] == 1:
        one_hot_label[i][1] = 1
      if num_examples[i] == 2:
        one_hot_label[i][2] = 1
      if num_examples[i] == 3:
        one_hot_label[i][3] = 1
    print(one_hot_label.type())
    print(pred_rot.type())
    pred = torch.matmul(one_hot_label, torch.t(pred_rot))
    result = torch.sum(pred,dim=1)
    G_rot_loss = torch.mean(result)
    G_loss = -alpha * G_rot_loss + G_loss

    G_loss.backward(retain_graph=True)
    optimizer_G.step()

    # ------------------------discriminator training--------------------------

    optimizer_D.zero_grad()

    '''
    calculate discriminator loss : hinge loss
    '''
    pred_real_logits = discriminator(real_data)[1]
    #D_loss_real = get_d_loss(prediction_real, ones_target(N))

    pred_fake_logits = discriminator(fake_data)[1]
    #D_loss_fake = loss(prediction_fake,zeros_target(N))
    D_loss = get_d_loss(pred_real_logits, pred_fake_logits)

    '''
                Compute rotation loss
                1. generate images with rotations : 0 90 180 270
                2. compute loss
    '''
    real_data90 = torch.rot90(real_data, 3, [2,3])
    real_data180 = torch.rot90(real_data, 2, [2,3])
    real_data270 = torch.rot90(real_data,1, [2,3])
    real_data_rot = torch.cat((real_data, real_data90, real_data180, real_data270),0)
    _,_,h1,_ = discriminator(real_data_rot)
    pred_rot = torch.log(h1 + 1e-10)

    pred = torch.matmul(one_hot_label, torch.t(pred_rot))
    result = torch.sum(pred,dim=1)
    D_rot_loss = torch.mean(result)
    D_loss = -beta * D_rot_loss + D_loss

    D_loss.backward(retain_graph=True)
    optimizer_D.step()

    # print loss and accuracy
    if nth_batch % 100 == 0:
      print('Training epoch: {} [{}/{} ({:.0f}%)]\t Training Loss: {:.6f}'.format(
        epoch, nth_batch * len(real_batch), len(train_loader.dataset),
        100. * nth_batch / len(train_loader), D_loss.item())
      )

