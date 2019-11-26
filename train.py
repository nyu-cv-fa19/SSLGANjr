import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from model import Generator, Discriminator

# define loss
loss = torch.nn.BCELoss()

# define rotation loss
weight_rotation_loss_d = 1 
weight_rotation_loss_g = 0.2

# generate noise 
noise_size = 128
def noise(size):
	all = []
	mu = 0
	sigma = 1
	for i in range(64):
		n = torch.Tensor(np.random.normal(0,1,size))
		n = Variable(n)
		all.append(n)
	all = Variable(all)
	return n

# generate label 
def ones_target(size):
	data = Variable(torch.ones(size,1))
	return data

def zeros_target(size):
	data = Variable(torch.zeros(size,0))
	return data

def generate_rotate_label(size):
	target = []
	for i in range(4):
		target = target + [i]*size
	return target

alpha = 0.2
beta = 1

batch_size = 64
num_examples = 64
ROTATE_NUM = 4 

# get discriminator obj and generator obj
discriminator = Discriminator()
generator = Generator()

# optimizer
optimizer_G = torch.optim.Adam(discriminator.parameters(), lr = 0.0002)
optimizer_D = torch.optim.Adam(generator.parameters(), lr = 0.0002)


# training
for epoch in range(epochs):

	for nth_batch, (real_batch,_) in enumerate(dataloader):
		
		N = real_batch.size(0)

        # prepare real data and fake data
		real_data = Variable(real_batch)
		fake_data = generator(noise(N))

		# ---------------------------generator training--------------------------------
		optimizer_G.zero_grad()

		# true/false loss for G
		G_loss = loss(discriminator(fake_data)[0], zeros_target(N))
        
        '''
        Compute rotation loss
        1. generate images with rotations : 0 90 180 270
        2. compute loss  
		'''
		fake_data90 = torch.rot90(fake_data, 3, [2,3])
		fake_data180 = torch.rot90(fake_data, 2, [2,3])
		fake_data270 = torch.rot90(fake_data,1, [2,3])
		fake_data_rot = torch.cat((fake_data, fake_data90, fake_data180, fake_data270),0)
        
		num_examples = generate_rotate_label(64) #[0,0,0,0,1,1,1,1,2,2,2,2]
		pred_rot = torch.log(discriminator(fake_data_rot)[2])

		# generate one hot vector according to rot label
		one_hot_label = torch.zeros([64*4,ROTATE_NUM], dtype = torch.int32)
		for i in range(64*4):
			if num_examples[i] == 0:
				one_hot_label[i][0] = 1
			if num_examples[i] == 1:
				one_hot_label[i][1] = 1
			if num_examples[i] == 2:
				one_hot_label[i][2] = 1
			if num_examples[i] == 3:
				one_hot_label[i][3] = 1

		pred = torch.matmul(one_hot_label, pred_rot) 
		result = torch.sum(pred,dim=1)
		G_rot_loss = torch.mean(result) 
		G_loss = alpha * G_rot_loss + G_loss

		G_loss.backward()
		optimizer_G.step()
		
		# ------------------------discriminator training--------------------------

		optimizer_D.zero_grad()
        
		prediction_real = discriminator(real_data)[0]
		D_loss_real = loss(prediction_real, ones_target(N))

		prediction_fake = discriminator(fake_data)[0]
		D_loss_fake = loss(prediction_fake,zeros_target(N))
		D_loss = D_loss_real + D_loss_fake

		'''
        Compute rotation loss
        1. generate images with rotations : 0 90 180 270
        2. compute loss  
		'''
		real_data90 = torch.rot90(real_data, 3, [2,3])
		real_data180 = torch.rot90(real_data, 2, [2,3])
		real_data270 = torch.rot90(real_data,1, [2,3])
		real_data_rot = torch.cat((real_data, real_data90, real_data180, real_data270),0)
		pred_rot = discriminator(real_data_rot)[2]

		pred = torch.matmul(one_hot_label, pred_rot) 
		result = torch.sum(pred,dim=1)
		D_rot_loss = torch.mean(result) 
		D_loss = beta * D_rot_loss + D_loss

		D_loss.backward()
        optimizer_D.step()

        # print loss and accuracy
        if nth_batch % 100 == 0:
        	print('Training epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        		epoch, nth_batch * len(real_batch), len(train_loader.dataset),
        		100. * nth_batch / len(train_loader), D_loss.item() 
        		)
        	)




