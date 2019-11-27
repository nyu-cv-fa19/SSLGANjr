import torch
import torch.nn as nn
import torch.nn.funcitonal as F
from resblock import G_ResidualBlock

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
                x = x.view(-1, 256 * self.s * self.s)
		x = self.res1(x)
		x = self.res2(x)
		x = self.res3(x)
		x = self.bn(x)
		x = self.relu(x)
		x = self.conv(x)
		x = self.tahn(x)
		return x
