import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from model import Generator, Discriminator
from data import train_loader, dataset,trainset
import torch.nn.functional as F
from ops import noise, generate_rotate_label,get_d_loss, get_g_loss
from torch.autograd import grad 

# check cuda availability
if torch.cuda.is_available():
  print('running on cuda')
else:
  print('running on cpu')

def calc_gp(real_data, fake_data, batch_size):
  
  # Calculate interpolation
  t = torch.rand(batch_size, 1, 1, 1).expand_as(real_data)
  if torch.cuda.is_available():
    t = t.cuda()

  interp = Variable(t * real_data.data + (1 - t) * fake_data.data, requires_grad=True)
  if torch.cuda.is_available():
    interp = interp.cuda()

  _, prob_interpolated, _, _ = discriminator(interp)

  if torch.cuda.is_available():
    gradients = grad( outputs=prob_interpolated, 
                      inputs=interp,
                      grad_outputs=torch.ones(prob_interpolated.size()).cuda(),
                      create_graph=True, retain_graph=True)[0]
  else:
    gradients = grad( outputs=prob_interpolated, 
                    inputs=interp,
                    grad_outputs=torch.ones(prob_interpolated.size()),
                    create_graph=True, retain_graph=True)[0]

  gradients = gradients.view(batch_size, -1)
  gradients_norm = torch.sqrt(torch.sum(torch.pow(gradients,2), dim=1) + 1e-10)
  return weight * torch.pow(gradients_norm - 1,2).mean()

epochs = 100
# define weights for rotation loss 
weight_rotation_loss_d = 1
weight_rotation_loss_g = 0.2
weight = 10

# noise size 
noise_size = 128

alpha = 0.2
beta = 1
betas = (0.9,0.99)
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
betas = (0.9,0.99)
optimizer_G = torch.optim.Adam(discriminator.parameters(), lr = 0.0002, betas = betas)
optimizer_D = torch.optim.Adam(generator.parameters(), lr = 0.0002, betas = betas)

# training
for epoch in range(epochs):
  
  train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

  for nth_batch, (real_batch,_) in enumerate(train_loader):

    N = real_batch.size(0)

    # prepare real data and fake data
    real_data = Variable(real_batch)
    fake_data = generator(noise(noise_size,N))

    real_data90 = torch.rot90(real_data, 3, [2,3])
    real_data180 = torch.rot90(real_data, 2, [2,3])
    real_data270 = torch.rot90(real_data,1, [2,3])
    real_data = torch.cat((real_data, real_data90, real_data180, real_data270),0)
    
    fake_data90 = torch.rot90(fake_data, 3, [2,3])
    fake_data180 = torch.rot90(fake_data, 2, [2,3])
    fake_data270 = torch.rot90(fake_data,1, [2,3])
    fake_data = torch.cat((fake_data, fake_data90, fake_data180, fake_data270),0)


    if torch.cuda.is_available():
      real_data = real_data.cuda()
      fake_data = fake_data.cuda()

    # ------------------------discriminator training--------------------------

    optimizer_D.zero_grad()

    _,d_pred_real_logits, d_pred_real_prob,_ = discriminator(real_data)
    _,d_pred_fake_logits,_,_ = discriminator(fake_data)
    
    gp = calc_gp(real_data[:N], fake_data[:N], N)
    D_loss = get_d_loss(d_pred_real_logits[:N], d_pred_fake_logits[:N]) + gp

    '''
                Compute rotation loss
                1. generate images with rotations : 0 90 180 270
                2. compute loss
    '''
    
    # generate one hot vector according to rot label
    num_examples = generate_rotate_label(N) #[0,0,0,0,1,1,1,1,2,2,2,2]
    one_hot_label = torch.zeros([N*4,ROTATE_NUM], dtype = torch.float32)
    for i in range(N*4):
      if num_examples[i] == 0:
        one_hot_label[i][0] = 1
      if num_examples[i] == 1:
        one_hot_label[i][1] = 1
      if num_examples[i] == 2:
        one_hot_label[i][2] = 1
      if num_examples[i] == 3:
        one_hot_label[i][3] = 1
    if torch.cuda.is_available():
      one_hot_label = one_hot_label.cuda()

    pred_rot = torch.log(d_pred_real_prob + 1e-10)
    pred = one_hot_label * pred_rot
    result = torch.sum(pred,dim=1)
    D_rot_loss = -torch.mean(result)
    D_loss = beta * D_rot_loss + D_loss

    D_loss.backward(retain_graph=True)
    optimizer_D.step()

    # print loss and accuracy
    if nth_batch % 100 == 0:
      print('Training epoch: {} [{}/{} ({:.0f}%)]\t Discriminator Loss: {:.6f}'.format(
        epoch+1, nth_batch * len(real_batch), len(train_loader.dataset),
        100. * nth_batch / len(train_loader), D_loss.item())
      )


    # ---------------------------generator training--------------------------------
    optimizer_G.zero_grad()
    
    # true/false loss for G
    _,g_pred_fake_logits,g_pred_real_prob,_ = discriminator(fake_data)
    
    G_loss = get_g_loss(g_pred_fake_logits[:N])

    num_examples = generate_rotate_label(N) #[0,0,0,0,1,1,1,1,2,2,2,2]
    # generate one hot vector according to rot label
    one_hot_label = torch.zeros([N*4,ROTATE_NUM], dtype = torch.float32)
    for i in range(N*4):
      if num_examples[i] == 0:
        one_hot_label[i][0] = 1
      if num_examples[i] == 1:
        one_hot_label[i][1] = 1
      if num_examples[i] == 2:
        one_hot_label[i][2] = 1
      if num_examples[i] == 3:
        one_hot_label[i][3] = 1
    if torch.cuda.is_available():
      one_hot_label = one_hot_label.cuda()
    
    pred_rot = torch.log( g_pred_real_prob + 1e-10)
    pred = one_hot_label * pred_rot
    result = torch.sum(pred,dim=1)
    G_rot_loss = -torch.mean(result)
    G_loss = alpha * G_rot_loss + G_loss

    G_loss.backward(retain_graph=True)
    optimizer_G.step()

    if nth_batch % 100 == 0:
      print('Training epoch: {} [{}/{} ({:.0f}%)]\t Generator Loss: {:.6f}'.format(
        epoch+1, nth_batch * len(real_batch), len(train_loader.dataset),
        100. * nth_batch / len(train_loader), G_loss.item())
      )
      print('\n')


