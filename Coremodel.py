import torch.nn as nn
import pdb

class Flatten(nn.Module):
    def forward(self,x):
        return x.view(x.shape[0],-1)



class coreModel(nn.Module):
    def __init__(self,):
        super(coreModel,self).__init__()
        # nn.Conv2d(in_channel,out_channel,kernel_size,padding)

        self.conv1 = nn.Conv2d(3,6,3,padding = 1) 
        self.conv2 = nn.Conv2d(6,12,3,padding = 1) 
        self.conv3 = nn.Conv2d(12,24,3,padding = 1) 
        self.fc = nn.Sequential(
                Flatten(),
                nn.Linear(4*4*24,24),
                nn.Linear(24,10)
        )

        self.activate = nn.ReLU()
        #self.activate = nn.ReLU()

    def forward(self,x):
        #x : 513
        x = self.activate(self.conv1(x)) # 256
        x = self.activate(self.conv2(x)) # 128
        x = self.activate(self.conv3(x)) # 64

        x = self.fc(x)

        return x
