import torch.nn as nn

class Denoise_Autoencoder_channel(nn.Module):
    def __init__(self,):
        super(Denoise_Autoencoder_channel,self).__init__()
        # nn.Conv2d(in_channel,out_channel,kernel_size,padding)

        self.conv1 = nn.Conv2d(3,6,3,padding = 1) 
        self.conv2 = nn.Conv2d(6,12,3,padding = 1) 
        self.conv3 = nn.Conv2d(12,24,3,padding = 1) 
        self.conv4 = nn.Conv2d(24,48,3,padding = 1) 
        self.conv5 = nn.Conv2d(48,96,3,padding = 1) 
       
        self.deconv1 = nn.Conv2d(96,48,3,padding = 1)
        self.deconv2 = nn.Conv2d(48,24,3,padding = 1)
        self.deconv3 = nn.Conv2d(24,12,3,padding = 1) 
        self.deconv4 =nn.Conv2d(12,6,3,padding = 1) 
        self.deconv5 = nn.Conv2d(6,3,3,padding = 1)

        self.activate = nn.ELU()
        self.regression = nn.Sigmoid() 
        #self.activate = nn.ReLU()

    def forward(self,x):
        #x : 513
        out_1 = self.activate(self.conv1(x)) # 256
        out_2 = self.activate(self.conv2(out_1)) # 128
        out_3 = self.activate(self.conv3(out_2)) # 64
        out_4 = self.activate(self.conv4(out_3)) # 32
        out_5 = self.activate(self.conv5(out_4)) # 16
        
        out = out_5

        out = self.activate(self.deconv1(out))
        out = self.activate(self.deconv2(out))
        out = self.activate(self.deconv3(out))
        out = self.activate(self.deconv4(out))
        
        #out = (self.regression(self.deconv5(out))-0.5)/0.224
        out = self.regression(self.deconv5(out))
        return out
