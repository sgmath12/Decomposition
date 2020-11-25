import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.models as models
import datasets
from Coremodel import *
from tqdm import tqdm
import pdb
import tensorly as tl
from tensorly.decomposition import parafac
from tensorly.decomposition import tucker
import torchattacks

def tucker_decomposition(X):
    N,C,H,W = X.shape
    rank = 4
    tucker_rank = [C,rank,rank]
    Tucker_reconstructions = torch.zeros_like(X).cpu()
    Cores = torch.zeros([N,C,rank,rank])
    for j,img in enumerate(X):
        core,tucker_factors = tucker(img,ranks = tucker_rank,init = 'random', tol = 1e-4,  random_state=np.random.RandomState())
        tucker_reconstruction = tl.tucker_to_tensor((core,tucker_factors))
        Tucker_reconstructions[j] = tucker_reconstruction
        Cores[j] = core

    return Cores
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier = torch.load('./saved/resnet50_cifar10.pth')
num_classes = 10

model = coreModel()

tl.set_backend('pytorch') 
batch_size = 64
train_set,test_set = datasets.CIFAR10(root = '/home/taejoon/data/CIFAR10',normalize = False)
train_loader = torch.utils.data.DataLoader(train_set,batch_size = batch_size,shuffle = True)
test_loader =  torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
model.eval()
model = model.to(device) 
running_loss = 0.0
attack= torchattacks.PGD(classifier, eps= float(8/255.0))

train_loss_epoch = []
train_acc_epoch = []
epoches = 50
for epoch in range(1,epoches):
    total_loss = 0
    total_acc = 0

    for (X,y) in tqdm(train_loader):
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()

        try :
            core = tucker_decomposition(X)
            core = core.to(device)
            
            preds = model(core)
            loss = criterion(preds,y)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

            _,prediction = preds.max(dim = 1,keepdim = False)
            total_acc += (prediction == y).sum()
            total_loss += loss
        except :
            None
            
    total_loss = total_loss / len(train_set)
    total_acc = total_acc.item()/ len(train_set)
    print ("loss",total_loss, "acc",total_acc)
    train_loss_epoch.append(100*total_loss)
    train_acc_epoch.append(total_acc)

    if epoch % 5 == 0 :
        torch.save(model,'./cifar10_models/resnet50_tucker_' +str(epoch) + '.pth')
torch.save(model,'./cifar10_models/resnet50_tucker_final'+ '.pth')
