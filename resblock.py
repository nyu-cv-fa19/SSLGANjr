import torch
import torch.nn as nn
import torch.nn.function as F

'''
create basic residual block x + F(x)

'''
class G_ResidualBlock(nn.Module):

	def __init__(self,
		         in_channels, 
		         out_channels, 
		         stride=1,
		         kernel_size = 3,
		         spectral_norm = False 
		         upsample=None):

		super(G_ResidualBlock,self).__init__()

        self.upsample = upsample

        # add spectral norm

		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,padding=1)
		self.bn1 = nn.BatchNorm2d(in_channels)
		self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(out_channels)
		self.upsample = nn.Upsample(scale_factor = 2, mode = 'near')


	def forward(self,x):
		
		residual = x
		x = self.bn1(x)
		x = F.relu(x)
		x = self.conv1(x)
		x = self.bn2(x)
		x = F.relu(x)
		x = self.conv2(x)
		# if downsample needed, do downsample to resize x 
		if self.upsample:
			residual = self.upsample(x)
		x = x + residual
		x = F.relu(x)
		return x






'''
class Resnet(nn.Module):

	def __init__(self,block, num_blocks, num_classes=10):

		super(Resnet,self).__init__()

		self.in_channels = 16 

		self.conv1 = nn.Conv2d(3,16,kernel_size=3,stride=1,padding=1)
		self.bn1 = nn.BatchNorm2d(16)
        
        # add 3 kinds of residual layers
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1],2)
        self.layer3 = self.make_layer(block, 64,layers[2],2)

        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64,num_classes)

    
	def make_layer(self, block, out_channels, num_blocks,stride=1):	

		downsample = None
		
		# if stride !=1 => size of x and that F(x) are not different 
		if(stride!=1) or (self.in_channels != out_channels):
			downsample = nn.Sequential(
				conv1(self.in_channels, out_channels, stride=stride),
				nn.BatchNorm2d(out_channels))

		# make layers 
		layers = []
		layers.append(block(self.in_channels,out_channels,stride,downsample))
		self.in_channels = out_channels
		for i in range(1,blocks):
			layers.append(block(out_channels,out_channels))
			return nn.Sequential(*layers)


	def forward(self,x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = F.relu(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)

		x = self.avg_pool(x)
		x = x.view(,-1)
		x = self.fc(x)
		return x


model = Resnet(ResidualBlock,[2,2,2])

'''


