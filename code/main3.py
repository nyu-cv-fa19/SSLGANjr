'''
new loss: 
minimize the distance of prob of different rots between real and fake
'''
import torch
import torch.optim as optim
from dataloaders import get_mnist_dataloaders, get_cifar_dataloaders
from model import Generator
from model import Discriminator
from training3 import Trainer


data_loader, _ = get_cifar_dataloaders(batch_size = 64)


inchannels = 3
generator = Generator(128,inchannels,32)
discriminator = Discriminator(inchannels)

if torch.cuda.is_available():
    generator = generator.cuda()
    discriminator = discriminator.cuda()


lr = 1e-4
betas = (.9, .99)
G_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=betas)
D_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

epochs = 200
trainer = Trainer(generator, discriminator, G_optimizer, D_optimizer)
trainer.train(data_loader, epochs)

# save generator and discriminator 
dataset = 'cifar'
torch.save(trainer.G.state_dict(), './new_loss_G_' + dataset + '.pt')
torch.save(trainer.D.state_dict(), './new_loss_D_' + dataset + '.pt')

