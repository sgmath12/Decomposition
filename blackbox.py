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
from utils import *



class blackbox(nn.Module):
    
    def __init__(self, model, Autoencoder = None, normalize = False):
        super(blackbox, self).__init__()
        self.model = model
        self.ae = Autoencoder
        self.normalize = normalize

    def forward(self, imgs ):
        # use Autoencoder
        if self.ae != None :
            imgs = self.ae(imgs)
        if self.normalize == True :
            self.model = nn.Sequential(Normalize(CIFAR_MEAN, CIFAR_STD), self.model)
        return self.model(imgs)



def main(args):
    classifier = torch.load('./saved/resnet50_cifar10.pth')
    model = blackbox(classifier)
    train_set,test_set = datasets.CIFAR10(root = '/home/taejoon/data/CIFAR10',normalize=False)
    num_classes = 10
    batch_size = 64

    test_loader = torch.utils.data.DataLoader(test_set,batch_size = batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    if args.attack =='FGSM':
        attack = torchattacks.FGSM(model, eps= float(Fraction(args.eps)))
    elif args.attack =='PGD':
        attack = torchattacks.PGD(model, eps= float(Fraction(args.eps)))
    elif args.attack =='DeepFool':
        attack = torchattacks.DeepFool(model)
    elif args.attack =='CW':
        attack = torchattacks.CW(model,kappa = 50)
    elif args.attack =='EOT':
        attack = torchattacks.APGD(model, eps= float(Fraction(args.eps)))
    # print (dict(torchattacks))

    model = model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0
    clean_acc = 0
    adv_acc = 0
    cp_acc = 0
    tucker_acc = 0 
    cp_rank = int(args.cp_rank)
    tucker_rank = [3,int(args.tucker_rank),int(args.tucker_rank)]

    for i,(images,labels) in enumerate(test_loader):
        
        images = images.to(device)
        labels = labels.to(device)

        preds = model(images)
        _,prediction = preds.max(dim = 1,keepdim = False)
        clean_acc += (prediction == labels).sum()
        adversarial_images = attack(images, labels)
        
        preds = model(adversarial_images)
        _,prediction = preds.max(dim = 1,keepdim = False)    
        adv_acc += (prediction == labels).sum()
        
        adversarial_images = adversarial_images.detach().cpu().numpy()
        # Cp_reconstructions = np.zeros_like(adversarial_images)
        Tucker_reconstructions = np.zeros_like(adversarial_images)

        for j,adv in enumerate(adversarial_images):
            # factors = tl.decomposition.parafac(adv,rank = cp_rank,init = 'random',tol = 1e-4,random_state = np.random.RandomState())
            # cp_reconstruction = tl.kruskal_to_tensor(factors)
            # Cp_reconstructions[j] = cp_reconstruction
            

            core,tucker_factors = tucker(adv,ranks = tucker_rank,init = 'random', tol = 1e-4,  random_state=np.random.RandomState())
            tucker_reconstruction = tl.tucker_to_tensor((core,tucker_factors))
            Tucker_reconstructions[j] = tucker_reconstruction

        # Cp_reconstructions = torch.from_numpy(Cp_reconstructions).to(device,dtype = torch.float)
        # preds = model(Cp_reconstructions)
        # _,prediction = preds.max(dim = 1,keepdim = False)

        # cp_acc += (prediction == labels).sum()

        Tucker_reconstructions = torch.from_numpy(Tucker_reconstructions).to(device,dtype = torch.float)
        preds = model(Tucker_reconstructions)
        _,prediction = preds.max(dim = 1,keepdim = False)

        tucker_acc += (prediction == labels).sum()
        # if (i%25==0):
        #     print ("# data %f, clean_acc %f, adv_acc %f, cp_acc %f, tucker_acc %f" %((i+1)*batch_size,clean_acc,adv_acc,cp_acc,tucker_acc))
        if (i%5==0):
            print ("# data %f, clean_acc %f, adv_acc %f, tucker_acc %f" %((i+1)*batch_size,clean_acc,adv_acc,tucker_acc))
        
    clean_acc = float(clean_acc) / len(test_set)
    adv_acc = float(adv_acc)/ len(test_set)
    # cp_acc = float(cp_acc)/len(test_set)
    tucker_acc = float(tucker_acc)/len(test_set)

    print ("loss",total_loss, "adv acc",adv_acc, "tucker acc",tucker_acc)
    f = open("./result/20200706.txt",'a')

    data = "Attack Method : " + args.attack + "\n"
    data += "epsilon : " + args.eps + "\n"
    data += "cp_rank : " + str(cp_rank) + ", tucker_rank : " + str(tucker_rank) + "\n"
    # data += "clean : " + str(clean_acc) + " adv_acc : " + str(adv_acc) + " cp_acc : " + str(cp_acc) + " tucker_acc : " + str(tucker_acc) + "\n"
    data += "clean : " + str(clean_acc) + " adv_acc : " + str(adv_acc) + " cp_acc : " + "None" + " tucker_acc : " + str(tucker_acc) + "\n"
    data += "="*50
    f.write(data + '\n')

if __name__ == "__main__":
        parser = argparse.ArgumentParser()

        parser.add_argument('--cp_rank',default = 1)
        parser.add_argument('--tucker_rank',default = 1)
        parser.add_argument('--eps',default = 1/255.0)
        parser.add_argument('--attack', default = 'FGSM')
        args = parser.parse_args()

        main(args)
