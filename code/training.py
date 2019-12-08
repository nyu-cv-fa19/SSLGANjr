import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import torch.nn.functional as F
import numpy as np
from utils import noise, generate_rotate_label,get_d_loss, get_g_loss


class Trainer():

    def __init__(self, generator, discriminator, 
                 optimizer_G, optimizer_D,
                 beta = 1, alpha =0.2):

        self.G = generator
        self.D = discriminator
        self.G_opt = optimizer_G
        self.D_opt = optimizer_D
        self.losses = {'G': [], 'D': []}
        self.beta = beta
        self.alpha = alpha


    def train_discriminator(self, real_data, fake_data, batch_size):
        
        real_data = Variable(real_data)
        
        # real or fake loss
        _, d_real_logits, d_real_rot_prob, _ = self.D(real_data)
        _, g_fake_logits, g_fake_rot_prob, _ = self.D(fake_data)
        
        # calculate gradient penalty loss
        gp = self.calc_gp(real_data[:batch_size], fake_data[:batch_size])

        self.D_opt.zero_grad()
       
        d_loss = get_d_loss(d_real_logits[:batch_size], g_fake_logits[:batch_size]) + gp
       
        # generate rotated data label
        num_examples = generate_rotate_label(batch_size) #[0,0,0,0,1,1,1,1,2,2,2,2]
        one_hot_label = torch.zeros([batch_size*4,4], dtype = torch.float32)
        for i in range(batch_size*4):
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
        
        # calculate rotation loss
        d_real_rot_loss = - torch.mean(torch.sum(one_hot_label * torch.log(d_real_rot_prob + 1e-12),dim=1))
        d_loss = self.beta * d_real_rot_loss + d_loss
        d_loss.backward(retain_graph=True)

        self.D_opt.step()

        # append loss
        self.losses['D'].append(d_loss.data)


    def train_generator(self, fake_data, batch_size):
        self.G_opt.zero_grad()

        #calculate real or fake loss
        _, g_fake_logits, g_fake_rot_prob, _ = self.D(fake_data)
        g_loss = get_g_loss(g_fake_logits[:batch_size])
      
        # generate label for rotation data
        num_examples = generate_rotate_label(batch_size) #[0,0,0,0,1,1,1,1,2,2,2,2]
        one_hot_label = torch.zeros([batch_size*4,4], dtype = torch.float32)
        for i in range(batch_size*4):
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
   
        # calculate rotation loss
        g_fake_rot_loss = -torch.mean(torch.sum(one_hot_label * torch.log(g_fake_rot_prob + 1e-12),dim=1))
        
        g_loss = g_loss + self.alpha * g_fake_rot_loss

        g_loss.backward(retain_graph=True)
        self.G_opt.step()

        # append loss
        self.losses['G'].append(g_loss.data)


    def calc_gp(self, real_data, fake_data):
        
        batch_size = real_data.size()[0]
        # calculate weight
        t = torch.rand(batch_size, 1, 1, 1).expand_as(real_data)
        if torch.cuda.is_available():
            t = t.cuda()
        # interpolation
        interp = Variable(t * real_data.data + (1 - t) * fake_data.data, requires_grad = True)
        if torch.cuda.is_available():
            interp = interp.cuda()

        _, prob, _, _ = self.D(interp)
        # calculate gradient 
        gradients = torch_grad(outputs=prob, 
                               inputs=interp,
                               grad_outputs=torch.ones(prob.size()).cuda() if torch.cuda.is_available() else torch.ones(
                               prob.size()),
                               create_graph=True, retain_graph=True)[0]

        gradients = gradients.view(batch_size, -1)
        g_norm = torch.sqrt(torch.sum(torch.pow(gradients,2), dim=1) + 1e-12)
        weight = 10
        return weight * (torch.pow(g_norm - 1,2)).mean()

    def train_one_epoch(self, data_loader):
        for nth_batch, (real_batch, _ )in enumerate(data_loader):
       
            batch_size = real_batch.size()[0]
           
            # generate real and fake data
            real_data = Variable(real_batch)
            noise_vec = noise(128,batch_size)
            fake_data = self.G(noise_vec)
 
            # generate rotated real data
            real_data90 = torch.rot90(real_data, 3, [2,3])
            real_data180 = torch.rot90(real_data, 2, [2,3])
            real_data270 = torch.rot90(real_data,1, [2,3])
            real_data = torch.cat((real_data, real_data90, real_data180, real_data270),0)
            
            # generate rotated fake data
            fake_data90 = torch.rot90(fake_data, 3, [2,3])
            fake_data180 = torch.rot90(fake_data, 2, [2,3])
            fake_data270 = torch.rot90(fake_data,1, [2,3])
            fake_data = torch.cat((fake_data, fake_data90, fake_data180, fake_data270),0)

            if torch.cuda.is_available():
                real_data = real_data.cuda()
                fake_data = fake_data.cuda()
            
            self.train_discriminator(real_data, fake_data, batch_size)
            self.train_generator(fake_data, batch_size)

            if nth_batch % 10 == 0:
                print("[{}/{}]".format(nth_batch * len(real_batch),len(data_loader.dataset)))
                print("D: {}".format(self.losses['D'][-1]))
                print("G: {}".format(self.losses['G'][-1]))
            
    def train(self, data_loader, epochs):
 
        for epoch in range(epochs):
            print("\nEpoch {}".format(epoch + 1))
            self.train_one_epoch(data_loader)
            #if epoch % 20 == 0:
            #    name = str(epoch) + 'th'
            #    torch.save(trainer.G.state_dict(), './G_' + name + '.pt')
            #    torch.save(trainer.D.state_dict(), './D_' + name + '.pt') 
