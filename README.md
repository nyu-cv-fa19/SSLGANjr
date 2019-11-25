# SSL-GANjr

#### SRGAN + self supervised pretext task

Super-resolution Generative Adversarial Networks (SRGAN) is a generative adversarial network that can recover high-resolution single image from heavily downsampled input data, utilizing perceptual loss (consisted of adversarial loss and content loss). Within SRGAN there are two networks: generator enhances the resolution of the input low-resolution image, and discriminator distinguishes whether the input image is the output of the generator. 

In discriminator, self-supervised learning could be utilized to pretrain the model. The discriminator network could then be utilized to train other dataset for different downstream tasks, including rotation, jigsaw and so on.

Currently, most self-supervised learning involves single pretext task. Several papers try to combine multiple pretext tasks together to enhance the model. In this project, we try to apply multiple pretext tasks in pretrain stage. More research on this aspect will be done in the following week.

#### Downstream task
Task: semantic segmentation

According to Self-supervised Visual Feature Learning with Deep Neural Networks: A Survey by Longlong Jing and Yingli Tian, “However, no one tested the performance of the transferred learning on other tasks yet.” So we decided to try different applications and transfer the model learnt by self supervised learning to other tasks, such as classification, segmentation, etc.


#### Datasets and Network
Datasets: Cifar/KITTI
Network: VGG/ Resnet


