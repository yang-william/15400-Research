import torch
import torch.optim as optim
from loader import *
from model import *
from torch.utils.data import DataLoader
from utils import *
import torch.nn.functional as F

unet = U_net().cuda()
#checkpoint = torch.load('unetL1.pth')
#unet.load_state_dict(checkpoint['model_state_dict'])
validation_dataset = MicroscopyData('data',0,269,-2)
training_dataset = MicroscopyData('data',270,3809,-2)
valid_loader = DataLoader(validation_dataset, batch_size = 32, num_workers = 10)
train_loader = DataLoader(training_dataset, batch_size=32, shuffle=True, num_workers = 10)
train_opt = torch.optim.Adam(unet.parameters(), lr=0.001, betas=(0.5,0.999))
#train_opt.load_state_dict(checkpoint['optimizer_state_dict'])
critierion = torch.nn.L1Loss(reduction = 'sum')

reference2, reference = next(iter(valid_loader))
reference = reference[0:15]
reference2 = reference2[0:15]
result = [generate_grid(arr_to_2d(reference,5,i)) for i in range(32)]
write_ref(result,'unetL1v2','reference')

epochs = 100
initial = 0
#c=0

for epoch in range(initial,initial+epochs):
    refc = reference2.cuda()
    reference1 = unet(refc).cpu().detach().numpy()
    del refc
    result = [generate_grid(arr_to_2d(reference1,5,i)) for i in range(32)]
    write_ref(result,'unetL1v2',str(epoch))
    total_loss = 0
    total_loss2 = 0
    counter = 0
    for x,y in valid_loader:
	#print(3)
	x = x.cuda()
	y = y.cuda()
	y1 = unet(x)
	overest_loss = torch.sum(F.relu(y1 - y))
        loss = critierion(y,y1) 
        total_loss = total_loss + loss.item()
	total_loss2 = total_loss2 + overest_loss.item()
	counter += x.shape[0]
	#print(4)
	#print(c)
	#c=c+1
    loss_out = open('unetL1v2_valid_error.txt','a')
    loss_out.write('Epoch: %d, Error: %.10f, Overestimation Error: %.10f\n' % (epoch,total_loss/counter, total_loss2/counter))
    loss_out.close()

    counter = 0
    total_loss = 0
    total_loss2 = 0
    for x,y in train_loader:
	#print(1)
	x = x.cuda()
	y = y.cuda()
        y1 = unet(x)
	overest_loss = torch.sum(F.relu(y1 - y))
        loss = critierion(y,y1) 
	total_loss = total_loss + loss.item()
	total_loss2 = total_loss2 + overest_loss.item()
	loss = loss + overest_loss
        counter += x.shape[0]
        train_opt.zero_grad()
        loss.backward()
        train_opt.step()
	#print(2)
	#print(c)
	#c=c+1
    loss_out = open('unetL1v2_train_error.txt','a')
    loss_out.write('Epoch: %d, Error: %.10f, Overestimation Error: %.10f\n' % (epoch,total_loss/counter, total_loss2/counter))
    loss_out.close()
torch.save({
            'model_state_dict': unet.state_dict(),
	    'optimizer_state_dict': train_opt.state_dict(),
            }, 'unetL1v2.pth')
