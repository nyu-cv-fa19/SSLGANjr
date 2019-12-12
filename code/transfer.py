import torch
import torch.nn as nn
import torch.nn.functional as F
from resblock import G_ResidualBlock
from resblock import D_ResidualBlock
from __future__ import print_function
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from dataloaders import get_mnist_dataloaders, get_cifar_dataloaders

data_loader, _ = get_mnist_dataloaders(batch_size = 64)

class Net(nn.Module):

  def __init__(self, in_channels, out_channels):

    super(Net,self).__init__()

    self.in_channels = 1

    self.res1 = D_ResidualBlock(in_channels, 128, downsample=True, first_block=True)
    self.res2 = D_ResidualBlock(128, 128, downsample=True)
    self.res3 = D_ResidualBlock(128, 128)
    self.res4 = D_ResidualBlock(128, 128)

    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()
    self.softmax = nn.Softmax()

    self.fc1 = nn.Linear(128 ,1)
    self.fc2 = nn.Linear(128, 4)
    self.fc = nn.Linear(128, out_channels)

  def forward(self,x):
    x = self.res1(x)
    x = self.res2(x)
    x = self.res3(x)
    x = self.res4(x)
    x = self.relu(x)
    x = torch.sum(x, dim=(2, 3))
    x = self.fc(x)

    return self.softmax(x)

path = ''
newModel = Net()
for param in newModel.parameters():
	param.requires_grad = False

Dis = torch.load(path)
state_dict = newModel.load_state_dict(Dis)
new_state_dict = newModel.state_dict()
state_dict = {k: v for k, v in state_dict.items() if k in new_state_dict}
new_state_dict.update(state_dict)
newModel.load_state_dict(new_state_dict)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

def train(epoch):
	newModel.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = Variable(data), Variable(target)
		optimizer.zero_grad()
		output = newModel(data)
		loss = F.nll_loss(output, target)
		loss.backward()
		optimizer.step()
		if batch_idx%args.log_interval == 0:
			print('Train Epoch:{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))

for epoch in range(1, args.epochs + 1):
	train(epoch)
    model_file = 'model_' + str(epoch) + '.pth'
    torch.save(newModel.state_dict(), model_file)
    print('\nSaved model to ' + model_file + '. You can run `python evaluate.py --model' + model_file + '` to generate the Kaggle formatted csv file')
