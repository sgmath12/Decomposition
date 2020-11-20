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

class Cp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, imgs):
        # Tucker_reconstructions = np.zeros_like(imgs)
        Cp_reconstructions = torch.zeros_like(imgs).cpu()
        cp_rank = rank
        for j,img in enumerate(imgs):
            factors = tl.decomposition.parafac(img,rank = cp_rank,init = 'random',tol = 1e-4,random_state = np.random.RandomState())
            cp_reconstruction = tl.kruskal_to_tensor(factors)
            Cp_reconstructions[j] = cp_reconstruction

        
        # Tucker_reconstructions = torch.from_numpy(Tucker_reconstructions)

        return Cp_reconstructions

    @staticmethod
    def backward(ctx,grad_output):
        #BPDA-Identity
        grad_output = grad_output.to(device)
        return grad_output



class Tucker(torch.autograd.Function):
    @staticmethod
    def forward(ctx, imgs):
        # Tucker_reconstructions = np.zeros_like(imgs)
        Tucker_reconstructions = torch.zeros_like(imgs).cpu()
        tucker_rank = [3,rank,rank]
        for j,img in enumerate(imgs):
            core,tucker_factors = tucker(img,ranks = tucker_rank,init = 'random', tol = 1e-4,  random_state=np.random.RandomState())
            tucker_reconstruction = tl.tucker_to_tensor((core,tucker_factors))
            Tucker_reconstructions[j] = tucker_reconstruction

        
        # Tucker_reconstructions = torch.from_numpy(Tucker_reconstructions)

        return Tucker_reconstructions

    @staticmethod
    def backward(ctx,grad_output):
        #BPDA-Identity
        grad_output = grad_output.to(device)
        return grad_output

class DecompNet(nn.Module):
    
    def __init__(self, model, Autoencoder = None, decomposition = "tucker"):
        super(DecompNet, self).__init__()
        self.model = model
        self.ae = Autoencoder
        self.decomposition = decomposition
        self.device = device

    def forward(self, imgs ):
        # x = globals()[args.me_type].apply(input)
        imgs = Tucker.apply(imgs) if self.decomposition == "tucker" else Cp.apply(imgs)
        imgs = imgs.to(self.device)

        if self.ae != None :
            imgs = self.ae(imgs)

        return self.model(imgs)


def main(args):
    if args.autoencoder == 'True':
        autoencoder = torch.load('./saved/DAE_cifar10.pth')
    else :
        autoencoder = None

    classifier = torch.load('./saved/resnet50_cifar10.pth')
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model = DecompNet(classifier,Autoencoder = autoencoder,decomposition = args.decomposition)
    

    train_set,test_set = datasets.CIFAR10(root = '/home/taejoon/data/CIFAR10',normalize=False)
    num_classes = 10
    batch_size = 4
    test_loader = torch.utils.data.DataLoader(test_set,batch_size = batch_size)


    criterion = nn.CrossEntropyLoss()
    global rank
    rank = int(args.rank)


    tucker_rank = [3,rank,rank]
    total_loss = 0
    clean_acc = 0
    adv_acc = 0
    adv_clf_acc = 0
    cp_acc = 0
    tucker_acc = 0 

    tl.set_backend('pytorch') 

    if args.attack == "BPDA":
        attack= torchattacks.PGD(model, eps= float(Fraction(args.eps)))
    elif args.attack == "EoT":
        attack = torchattacks.APGD(model, eps= float(Fraction(args.eps)))

    model.eval()
    # attack_classifier = torchattacks.PGD(classifier,eps = float(Fraction(args.eps)))

    for i,(images,labels) in enumerate(test_loader):
        images,labels = images.to(device),labels.to(device)
        preds = model(images)
        _,prediction = preds.max(dim = 1,keepdim = False)
        clean_acc += (prediction == labels).sum()


        adversarial_images = attack(images, labels)
        preds = model(adversarial_images)
        _,prediction = preds.max(dim = 1,keepdim = False)    
        adv_acc += (prediction == labels).sum()


        # advs = attack_classifier(images,labels)
        # preds = model(advs)
        # _,prediction = preds.max(dim = 1,keepdim = False)  
        # adv_clf_acc += (prediction == labels).sum() 

        #
        # advs = advs.detach().cpu()
        # Tucker_reconstructions = torch.zeros_like(advs)
        # for j,adv in enumerate(advs):
        #     core,tucker_factors = tucker(adv,ranks = tucker_rank,init = 'random', tol = 1e-4,  random_state=np.random.RandomState())
        #     tucker_reconstruction = tl.tucker_to_tensor((core,tucker_factors))
        #     Tucker_reconstructions[j] = tucker_reconstruction

        # # Tucker_reconstructions = torch.from_numpy(Tucker_reconstructions).to(device,dtype = torch.float)
        # Tucker_reconstructions = Tucker_reconstructions.to(device)
        # preds2 = classifier(Tucker_reconstructions)
        # _,prediction = preds2.max(dim = 1,keepdim = False)

        # tucker_acc += (prediction == labels).sum()
        # pdb.set_trace()

        if (i%5==0):
            print ("method %s # data %f, clean_acc %f, adv_acc %f" %(args.decomposition,(i+1)*batch_size,clean_acc,adv_acc))
            
    clean_acc = float(clean_acc) / len(test_set)
    adv_acc = float(adv_acc)/ len(test_set)

    print ("method %s # data %f, clean_acc %f, adv_acc %f" %(args.decomposition,(i+1)*batch_size,clean_acc,adv_acc))
    print ("rank %s autoencoder %s"%(args.rank,args.autoencoder))
    f = open("./result/whitebox_1.txt",'a')

    data = "Attack Method : " + args.attack + "\n"
    data += "Decomposition method : " + args.decomposition + "\n"
    data +=  "Autoencoder : " + args.autoencoder + "\n"
    data += "epsilon : " + args.eps + "\n"
    data += "rank : " + str(rank) + "\n"
    data += "clean : " + str(clean_acc) + " adv_acc : " + str(adv_acc) + "\n"
    data += "="*50
    f.write(data + '\n')

if __name__ == "__main__":
        parser = argparse.ArgumentParser()

        parser.add_argument('--rank',default = 1)
        parser.add_argument('--decomposition',default = 'tucker')
        parser.add_argument('--eps',default = 1/255.0)
        parser.add_argument('--attack', default = 'BPDA')
        parser.add_argument('--autoencoder', default = 'False')
        args = parser.parse_args()

        main(args)