import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    class SubNet(torch.nn.Module):
        def __init__(self, f_in, f_out):
            super(Encoder.SubNet, self).__init__()
	    self.conv1 = nn.Conv3d(f_in, f_out, 5, 1, 2)
	    self.bn1 = nn.BatchNorm3d(f_out)
            self.conv = nn.Conv3d(f_out, f_out, 5, 4, 2)
            self.bn = nn.BatchNorm3d(f_out)
        def forward(self, x):
	    x = self.conv1(x)
	    x = self.bn1(x)
	    x = F.leaky_relu(x)
            x = self.conv(x)
            x = self.bn(x)
            x = F.leaky_relu(x)
            return x
        
    def __init__(self):
        super(Encoder, self).__init__()
	self.sn1 = self.SubNet(1,64)
	self.sn2 = self.SubNet(64,512)
        #self.sn1 = self.SubNet(1,32)
        #self.sn2 = self.SubNet(32,64)
        #self.sn3 = self.SubNet(64,128)
        #self.sn4 = self.SubNet(128,256)
        #self.sn5 = self.SubNet(256,512)
        #self.conv = nn.Conv2d(512,1024,2,2)
	self.conv = nn.Conv3d(512,1024,(2,4,4))
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.sn1(x)
        x = self.sn2(x)
        #x = self.sn3(x)
        #x = self.sn4(x)
        #x = self.sn5(x)
        #x = x.squeeze(2)
        x = self.conv(x)
	x = x.squeeze(2)#
        x = x.squeeze(2)
        x = x.squeeze(2)
        return x

class Decoder(nn.Module):
    class SubNet(torch.nn.Module):
        def __init__(self, f_in, f_out):
            super(Decoder.SubNet, self).__init__()
	    self.conv2 = nn.Conv3d(f_in, f_in, 5, 1, 2)
	    self.bn2 = nn.BatchNorm3d(f_in)
            self.conv = nn.ConvTranspose3d(f_in, f_out, 5, 4, 2, 2)
            self.bn = nn.BatchNorm3d(f_out)
	    self.conv1 = nn.Conv3d(f_out, f_out, 5, 1, 2)
	    self.bn1 = nn.BatchNorm3d(f_out)
        def forward(self, x):
            x = self.conv2(x)
            x = self.bn2(x)
            x = F.leaky_relu(x)
            x = self.conv(x)
            x = self.bn(x)
            x = F.leaky_relu(x)
            x = self.conv1(x)
            x = self.bn1(x)
            x = F.leaky_relu(x)
            return x

    def __init__(self):
        super(Decoder, self).__init__()
        #self.conv1 = nn.ConvTranspose2d(1024,512, 2, 2)
	self.conv1 = nn.ConvTranspose3d(1024,512,(2,4,4))
        self.bn1 = nn.BatchNorm3d(512)
        self.sn1 = self.SubNet(512,64)
        #self.sn2 = self.SubNet(256,128)
        #self.sn3 = self.SubNet(128,64)
        #self.sn4 = self.SubNet(64,32)
        self.conv3 = nn.ConvTranspose3d(64,1, 5, 4, 2, 2)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = x.unsqueeze(2)
        x = x.unsqueeze(2)
	x = x.unsqueeze(2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        #x = x.unsqueeze(2)
        x = self.sn1(x)
        #x = self.sn2(x)
        #x = self.sn3(x)
        #x = self.sn4(x)
        x = self.conv3(x)
        x = self.sigmoid(x)
        x = x.squeeze(1)
        return x

class U_net(nn.Module):
	class sub_net(nn.Module):
		def __init__(self, f_in, depth):
			super(U_net.sub_net, self).__init__()
			self.depth = depth
	    		self.conv1 = nn.Conv3d(f_in, f_in, 5, 1, 2)
	    		self.bn1 = nn.BatchNorm3d(f_in)
			if(depth == 0):
				return
            		self.conv2 = nn.Conv3d(f_in, f_in*2, 4, 2, 1)
			self.bn2 = nn.BatchNorm3d(f_in*2)
			self.snet = U_net.sub_net(f_in*2,depth -1)
			self.convt = nn.ConvTranspose3d(f_in*2, f_in, 4, 2, 1)
			self.bnt = nn.BatchNorm3d(f_in)
			self.conv3 = nn.Conv3d(f_in*2, f_in, 5, 1, 2)
			self.bn3 = nn.BatchNorm3d(f_in)

			
		def forward(self, x):
			x = self.conv1(x)
	    		x = self.bn1(x)
			#print(x.shape)
	    		x1 = F.leaky_relu(x)

			if(self.depth == 0):
				return x1
			
            		x = self.conv2(x1)
            		x = self.bn2(x)
			#print(x.shape)
            		x = F.leaky_relu(x)
			#print(x.shape)
			x = self.snet(x)
			#print(x.shape)
			x = self.convt(x)
			#print(x.shape)
			x = self.bnt(x)
	    		x = F.leaky_relu(x)
						
			x = torch.cat((x,x1),1)
			#print(x.shape)
			x = self.conv3(x)
			x = self.bn3(x)	
	    		x = F.leaky_relu(x)	
            		return x
			
		
	def __init__(self, f_in=32, depth=3):
	        super(U_net, self).__init__()
		self.conv1 = nn.Conv3d(13,13,5,1,2)
		self.bn1 = nn.BatchNorm3d(13)

		self.conv2 = nn.Conv3d(13, f_in, 4, 2, 1)	
		self.bn2 = nn.BatchNorm3d(f_in)

		self.snet = self.sub_net(f_in, depth)

		self.convt = nn.ConvTranspose3d(f_in, 13, 4, 2, 1)
		self.bnt = nn.BatchNorm3d(13)
		self.conv3 = nn.Conv3d(26, 1, 5, 1, 2)
		self.sigmoid = nn.Sigmoid()

	def forward(self,x):
		#print(x.shape)
		x = self.conv1(x)
		x = self.bn1(x)
		#print(x.shape)
		x1 = F.leaky_relu(x)

		x = self.conv2(x1)
		x = self.bn2(x)
		#print(x.shape)
		x = F.leaky_relu(x)
		x = self.snet(x)
		#print(x.shape)
		x = self.convt(x)
		x = self.bnt(x)
		x = F.leaky_relu(x)	   
		#print(x.shape)
		x = torch.cat((x,x1),1)
		#print(x.shape)
		x = self.conv3(x)

		#print(x.shape)
		x = self.sigmoid(x)
		x = x.squeeze(1)
		return x

class Discriminator(nn.Module):
    class SubNet(torch.nn.Module):
        def __init__(self, f_in, f_out):
            super(Discriminator.SubNet, self).__init__()
            self.conv = nn.Conv3d(f_in, f_out, 4, 2, 1)
            self.bn = nn.BatchNorm3d(f_out)
        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            x = F.leaky_relu(x)
            return x
        
    def __init__(self):
        super(Discriminator, self).__init__()
	self.sn1 = self.SubNet(14,32)
	self.sn2 = self.SubNet(32,64)
        #self.sn1 = self.SubNet(1,32)
        #self.sn2 = self.SubNet(32,64)
        self.sn3 = self.SubNet(64,128)
        self.sn4 = self.SubNet(128,256)
        #self.conv = nn.Conv2d(512,1024,2,2)
	self.conv = nn.Conv3d(256,1,(2,4,4))
	#self.sigmoid = nn.Sigmoid()
    def forward(self, x, y):
	y = y.unsqueeze(1)
	x = torch.cat([x,y],1)
	#print(x.shape)
        x = self.sn1(x)
	#print(x.shape)
        x = self.sn2(x)
	#print(x.shape)
        x = self.sn3(x)
	#print(x.shape)
        x = self.sn4(x)
        #x = x.squeeze(2)
	#print(x.shape)
        x = self.conv(x)
	#print(x.shape)
	#x = self.sigmoid(x)
	x = x.squeeze(2)#
        x = x.squeeze(2)
        x = x.squeeze(2)
	x = x.squeeze(1)
        return x

"""
from loader import *
dataset = MicroscopyData('data',0,0,0)
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=1) 
x, _ = next(iter(loader))
encoder = Encoder()
out = encoder(x)
decoder = Decoder()
x1 = decoder(out)
"""
