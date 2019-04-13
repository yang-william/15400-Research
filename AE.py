import torch
import torch.optim as optim
from loader import *
from model import *
from torch.utils.data import DataLoader
from utils import *

encoder = Encoder().cuda()
decoder = Decoder().cuda()
validation_dataset = MicroscopyData('data',0,269,0)
training_dataset = MicroscopyData('data',270,3809,0)
valid_loader = DataLoader(validation_dataset, batch_size = 32, num_workers = 10)
train_loader = DataLoader(training_dataset, batch_size=32, shuffle=True, num_workers = 10)
train_opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001, betas=(0.5,0.999))
critierion = torch.nn.MSELoss()

reference = next(iter(valid_loader))[0:15]
result = [generate_grid(arr_to_2d(reference,5,i)) for i in range(32)]
write_ref(result,'test5','reference')

epochs = 500
for epoch in range(epochs):
    reference1 = decoder(encoder(reference.cuda())).cpu().detach().numpy()
    result = [generate_grid(arr_to_2d(reference1,5,i)) for i in range(32)]
    write_ref(result,'test5',str(epoch))
    total_loss = 0
    for x in valid_loader:
	x = x.cuda()
        v = encoder(x)
        x1 = decoder(v)
        loss = critierion(x,x1)
        total_loss = total_loss + loss.item()
    loss_out = open('losses5.txt','a')
    loss_out.write('Epoch: %d, Error: %.7f\n' % (epoch,total_loss))
    loss_out.close()
    for x in train_loader:
	x = x.cuda()
        v = encoder(x)
        x1 = decoder(v)
        loss = critierion(x,x1)
        train_opt.zero_grad()
        loss.backward()
        train_opt.step()

