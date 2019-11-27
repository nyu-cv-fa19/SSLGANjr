import torch
import torch.nn as nn
import torch.nn.funcitonal as F
from resblock_d import D_ResidualBlock

class Discriminator(nn.Module):

	def __init__(self,block, in_channels, num_blocks):

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





