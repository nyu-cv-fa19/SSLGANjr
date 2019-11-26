import torch
import torch.nn as nn
import torch.nn.function as F
import torch.nn.utils.spectral_norm


'''
create basic residual block x + F(x)
'''
class D_ResidualBlock(nn.Module):

	def __init__(self, in_channels, out_channels, stride=1, downsample=None, first_block=False):

		super(D_ResidualBlock, self).__init__()

        self.downsample = downsample
        self.first_block = first_block

		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
		# self.bn1 = nn.BatchNorm2d(out_channels)
		self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3, stride=1, padding=1)
		self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
		# self.bn2 = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLu(inplace=True)
		self.avgpool = nn.AvgPool2d(2, 2, padding=1)


	def forward(self,x):
		residual = x
		if self.first_block:
			x = self.conv1(x)
			x = spectral_norm(x)
			x = self.relu(x)
			x = self.conv2(x)
			x = spectral_norm(x)
			x = self.relu(x)
		else:
			x = self.conv1(self.relu(x))
			x = spectral_norm(x)
			x = self.conv2(self.relu(x))
			x = spectral_norm(x)

		# if downsample needed, do downsample to resize x 
		if self.downsample is not None:
			x = self.avgpool(x)
			residual = self.avgpool(residual)
		
		residual = self.conv_shortcut(residual)
		x = x + residual

		return x


