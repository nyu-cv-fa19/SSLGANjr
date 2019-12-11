import torch
from utils import noise
import cv2

# reload generator model
def load_generator(filepath):
    model = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    for para in model.parameters():
        para.requires_grad = False
    
    model.eval()
    return model


filepath = '../results1/G_cifar.pt'
generator = load_generator(filepath)
num_data = len(test_loader.dataset)

# generate tensor and write to img
for i in range(num_data):
    output = generator(noise(128,1))
    print(output.size())
    fp = '../generate-img/' + str(i+1) + 'th_img' + '.png'
    output = output.numpy()
    cv2.imwrite(fp, output)

    