# SSLGAN with improved loss function

### Structure:
The code for the project is put in foler ```./code```
The .pth models for all experiments are in folder ```./code/result1```
FID scores are recorded in folder ```./log```
### How to run:

#### Train:
- Train GAN with SN: run ```main.py```  
- Tain SSLGAN: run ```main2.py```
- Train our improved SSLGAN: run ```main3.py```

#### Evaluation:
- Calculate FID:
  1. Generate images from trained generator
     1. Create folder `./generated-img` (generated images will be stored here)
     2. Sepcify which trained generater model will be used in the code [here](https://github.com/nyu-cv-fa19/SSLGANjr/blob/master/code/generate-img.py#L23)
     3. Run `generate-img.py`
  2. Extract test images from cifar10
     1. Create folder `./test-img` (real images will be stored here)
     2. Run `test-img.py`
  3. Calculate FID: run `python fid.py ./generated-img ./test-img`  
  To run the evaluation on GPU, use the flag `--gpu N`, where `N` is the index of the GPU to use.

- Transfer learning: cifar10 classification
  1. Open notebook `transfer.ipynb`
  2. Set discriminator to be used at the beginning of the 3rd cell `Dis = ...`  
  E.g., `Dis = torch.load("results1/D_200th.pt")`
  3. Run the notebook
