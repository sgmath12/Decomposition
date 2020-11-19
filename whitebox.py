import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data 
import datasets
import torchvision.models as models

import tensorly as tl
from tensorly.decomposition import parafac
from tensorly.decomposition import tucker
from tqdm import tqdm
import numpy as np
import pdb
import argparse

from advertorch.attacks import LinfPGDAttack
import torchattacks
from fractions import Fraction


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()

    def forward(self,x):
        x = x*2
        return x


class Tucker(torch.autograd.Function):
    @staticmethod
    def forward(ctx, imgs):
        # Tucker_reconstructions = np.zeros_like(imgs)
        Tucker_reconstructions = torch.zeros_like(imgs).cpu()
        
        for j,img in enumerate(imgs):
            core,tucker_factors = tucker(img,ranks = rank,init = 'random', tol = 1e-4,  random_state=np.random.RandomState())
            tucker_reconstruction = tl.tucker_to_tensor((core,tucker_factors))
            Tucker_reconstructions[j] = tucker_reconstruction

        
        # Tucker_reconstructions = torch.from_numpy(Tucker_reconstructions)

        return Tucker_reconstructions

    @staticmethod
    def backward(ctx,grad_output):
        #BPDA-Identity
        
        grad_output = grad_output.to(torch.device("cuda"))
        return grad_output

class DecompNet(nn.Module):
    
    def __init__(self, model, Autoencoder = None, decomposition = "tucker", device = None):
        super(DecompNet, self).__init__()
        self.model = model
        self.ae = Autoencoder
        self.decomposition = decomposition
        self.device = device

    def forward(self, imgs ):
        # x = globals()[args.me_type].apply(input)
        imgs = Tucker.apply(imgs) if self.decomposition == "tucker" else imgs
        imgs = imgs.to(self.device)
        
        imgs = self.ae(imgs) if self.ae != None else imgs
        return self.model(imgs)


model = torch.load('./saved/resnet50_cifar10.pth')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
decompnet = DecompNet(model,device = device)
decompnet.eval()

train_set,test_set = datasets.CIFAR10(root = '/home/taejoon/data/CIFAR10',normalize=False)
num_classes = 10
batch_size = 4
test_loader = torch.utils.data.DataLoader(test_set,batch_size = batch_size)


criterion = nn.CrossEntropyLoss()
global rank
rank = 4

tl.set_backend('pytorch') 

attack_decomp = torchattacks.FGSM(decompnet, eps= 1/255.0)
attack_model = torchattacks.FGSM(model, eps= 1/255.0)
model.eval()
for i,(images,labels) in enumerate(test_loader):
    images,labels = images.to(device),labels.to(device)
    # out_model = model(images)
    # images.requires_grad = True
    # out_decomp = decompnet(images)
    # cost_model = criterion(out_model,labels)
    # cost_decomp = criterion(out_decomp,labels)

    # grad = torch.autograd.grad(cost_decomp, images,retain_graph=False, create_graph=False)[0]
    adversarial_images = attack_decomp(images, labels)
    # pdb.set_trace()
    out_model = decompnet(images)
    out_adv = decompnet(adversarial_images)
    out = model(adversarial_images)
    pdb.set_trace()