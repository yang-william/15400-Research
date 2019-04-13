import torch
import torch.optim as optim
from loader import *
from model import *
from torch.utils.data import DataLoader
from utils import *
from torch.autograd import Variable

batch_size = 64

#unet = nn.DataParallel(U_net(16,3)).cuda()
#D = nn.DataParallel(Discriminator()).cuda()
unet = U_net(16,3).cuda()
D = Discriminator().cuda()
checkpoint = torch.load('pix2pix.pth')
unet.load_state_dict(checkpoint['model_state_dict'])
D.load_state_dict(checkpoint['model_d_state_dict'])
validation_dataset = MicroscopyData('data',0,269,-2)
training_dataset = MicroscopyData('data',270,3809,-2)

valid_loader = DataLoader(validation_dataset, batch_size = batch_size, num_workers = 10)
train_loader = DataLoader(training_dataset, batch_size= batch_size, shuffle=True, num_workers = 10)
train_opt = torch.optim.Adam(unet.parameters(), lr=0.001, betas=(0.5,0.999))
train_d_opt = torch.optim.Adam(D.parameters(), lr=0.001, betas=(0.5,0.999))
train_opt.load_state_dict(checkpoint['optimizer_state_dict'])
train_d_opt.load_state_dict(checkpoint['optimizer_d_state_dict'])
criterionL1 = torch.nn.L1Loss()
criterionGAN = nn.BCEWithLogitsLoss()

reference2, reference = next(iter(valid_loader))
reference = reference[0:15]
reference2 = reference2[0:15]
#result = [generate_grid(arr_to_2d(reference,5,i)) for i in range(32)]
#write_ref(result,'pix2pix','reference')

epochs = 50
initial = 25
#c=0

ones = torch.ones(batch_size).cuda()
zeros = torch.zeros(batch_size).cuda()
size = batch_size
for epoch in range(initial,initial+epochs):
    refc = reference2.cuda()
    reference1 = unet(refc).cpu().detach().numpy()
    del refc
    result = [generate_grid(arr_to_2d(reference1,5,i)) for i in range(32)]
    write_ref(result,'pix2pix',str(epoch))
    total_loss = 0
    total_D_loss = 0
    counter = 0
    for x,y in valid_loader:
	#print(3)
	x = x.cuda()
	y = y.cuda()
	y1 = unet(x)
	
	if(x.shape[0] != size):
		ones = torch.ones(x.shape[0]).cuda()
		zeros = torch.zeros(x.shape[0]).cuda()
		size = x.shape[0]

	D_real = criterionGAN(D(x,y),ones)

	D_fake = criterionGAN(D(x,y1),zeros)
	D_loss = (D_real + D_fake)*0.5
	total_D_loss += D_loss.item()
	del D_loss
	del D_real
	del D_fake
	G_fake = criterionGAN(D(x,y1),ones)
        loss = G_fake + 100*criterionL1(y,y1)
	total_loss = total_loss + loss.item()
	counter += x.shape[0]	
	del G_fake
	del loss
	
	#print(4)
	#print(c)
	#c=c+1
    loss_out = open('pix2pix_valid_error.txt','a')
    loss_out.write('[Epoch: %d] G_Error: %.10f, D_Error: %.10f \n' % (epoch,total_loss/counter,total_D_loss/counter))
    loss_out.close()

    counter = 0
    total_loss = 0
    total_D_loss = 0
    for x,y in train_loader:
	#print(1)
	#print(torch.cuda.memory_allocated())
	if(x.shape[0] != size):
		ones = torch.ones(x.shape[0]).cuda()
		zeros = torch.zeros(x.shape[0]).cuda()
		size = x.shape[0]
	x = x.cuda()
	y = y.cuda()  
	y1 = unet(x)    
	#print(2)  
	#print(torch.cuda.memory_allocated())
	D_real = criterionGAN(D(x,y),ones)

	D_fake = criterionGAN(D(x,y1),zeros)
	D_loss = (D_real + D_fake)*0.5
	#print(3)
	#print(torch.cuda.memory_allocated())
	train_d_opt.zero_grad()
	D_loss.backward()
	total_D_loss += D_loss.item()
	train_d_opt.step()
	#print(4)
	#print(torch.cuda.memory_allocated())
	temp = D_loss.item()
	del D_loss			
	del D_fake
	del D_real

	y1 = unet(x)
	#print(4)
	#print(torch.cuda.memory_allocated())
	G_fake = criterionGAN(D(x,y1),ones)
        loss = G_fake + 100*criterionL1(y,y1)
	#print(6)
	#print(torch.cuda.memory_allocated())
	total_loss = total_loss + loss.item()
        counter += x.shape[0]
        train_opt.zero_grad()
        loss.backward()
        train_opt.step()
	#print(7)
	#print(torch.cuda.memory_allocated())
	del G_fake
	
	print('[Epoch: %d] G_Error: %.10f, D_Error: %.10f \n' % (epoch,loss.item(),temp))
	del loss
	#print(2)
	#print(c)
	#c=c+1
    loss_out = open('pix2pix_train_error.txt','a')
    loss_out.write('[Epoch: %d] G_Error: %.10f, D_Error: %.10f \n' % (epoch,total_loss/counter,total_D_loss/counter))
    loss_out.close()
torch.save({
            'model_state_dict': unet.state_dict(),
	    'optimizer_state_dict': train_opt.state_dict(),
	    'model_d_state_dict': D.state_dict(),
	    'optimizer_d_state_dict': train_d_opt.state_dict(),
            }, 'pix2pix.pth')
