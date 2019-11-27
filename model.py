import torch
import torch.nn as nn
import torch.nn.funcitonal as F
from resblock import G_ResidualBlock
from resblock import D_ResidualBlock

class Generator(nn.Module):

	def __init__(self, z_size, output_channel, output_size = 32):

		super(Generator,self).__init__()

		# this is for cifar10 
		s = 4 
		self.output_size = output_size
		self.s = s
		self.z_size = z_size

		self.res1 = G_ResidualBlock(256,256, upsample = True)
		self.res2 = G_ResidualBlock(256,256, upsample = True)
		self.res3 = G_ResidualBlock(256,256, upsample = True)
		
                self.fc = nn.Linear(128,s*s*256)
		self.bn = nn.BatchNorm2d(256)
		self.relu = nn.ReLU()
		self.tanh = nn.Tanh()
		self.conv = nn.Conv2d(256,output_channel, padding = 1, kernel_size = 3, stride = 1)

	def forward(self,x):
		x = self.fc(x)
                x = x.view(-1, 256, self.s, self.s)
		x = self.res1(x)
		x = self.res2(x)
		x = self.res3(x)
		x = self.bn(x)
		x = self.relu(x)
		x = self.conv(x)
		x = self.tahn(x)
		return x


class Discriminator(nn.Module):

	def __init__(self, in_channels):

		super(Discriminator,self).__init__()

		self.in_channels = in_channels
		
		self.res1 = D_ResidualBlock(in_channels, 128, downsample=True, first_block=True)
		self.res2 = D_ResidualBlock(128, 128, downsample=True)
		self.res3 = D_ResidualBlock(128, 128)
		self.res4 = D_ResidualBlock(128, 128)
        
        self.relu = nn.ReLu()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

        self.fc1 = nn.Linear(128 ,1)
        self.fc2 = nn.Linear(128, 4)



	def forward(self,x):
		x = self.res1(x)
		x = self.res2(x)
		x = self.res3(x)
		x = self.res4(x)
		x = self.relu(x)
		x = torch.sum(x, dim=(2, 3))
		out_logit = self.fc1(x)
		out = self.sigmoid(out_logit)
		pre_logits = self.fc2(x)
		pre = self.softmax(pre_logits)
		
		return out, out_logit, pre, pre_logits
