# -*- coding: utf-8 -*- 
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torch.nn.functional as F
#import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision import datasets, transforms
from torch.autograd import Variable

batch_size = 200 #training batch_size
momentum = 0.5 #adam parameter
epochs = 500 # number of epochs (training)
lr = 0.0001 #learning rate
imageSize = 32 # resize input image to XX
nz = 100 #size of the latent z vector
ngf = 32 #number of G output filters
ndf = 32 #number of D output filters
nc = 3 # numbel of channel
outf = './fake' #folder to output images and model checkpoints
log_interval = 1000
test_interval = 5000
continue_netD = '' #"path to netD (to continue training"
continue_netG = '' #"path to netG (to continue training"
cudnn.benchmark = True
fsave = open('accuracy.txt','w')

dataloader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data', train=True, download=False,
                   transform=transforms.Compose([
                   	transforms.Scale(imageSize),
                   	transforms.ToTensor(),
                   	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   	]))
    ,batch_size=batch_size, shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data', train=True, download=True,
                   transform=transforms.Compose([
                    transforms.Scale(imageSize),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ]))
    ,batch_size=batch_size, shuffle=False, num_workers=2)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
#scp -r szhou@159.226.21.92:/data4/szhou/224dataset/Face/ ./224dataset

class _netG(nn.Module):
    def __init__(self):
        super(_netG, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            # ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
            # state size. (batch_size) x 100 x 1 x 1
            # h_o = (h_i + 2 * pad_h - kernel_h) / stride_h + 1  #conv layer
            # h_out=(h_in−1)∗stride[0]−2∗padding[0]+kernel_size[0]+output_padding[0]
            # w_out=(w_in−1)∗stride[1]−2∗padding[1]+kernel_size[1]+output_padding[1]
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 2, 1, bias=False), # h = (1-1) * 4 - 0 + 4 + 0 = 4 , w = 0 * 1 - 2 * 0 + 4 + 0 = 4
            #nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False), # h = (4-1) * 2 - 2*1 + 4 + 0 = 8
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        output = self.main(input)
        return output


netG = _netG()
#netG.apply(weights_init)



class _netD(nn.Module):
    def __init__(self):
        super(_netD, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
        )
        self.main2 = nn.Sequential(
            nn.Linear(1024,10),
        )
        self.main3 = nn.Sequential(
            nn.Sigmoid()
        )
    def forward(self, input,matching = False):
        output = self.main(input)
        feature = output.view(-1,1024)
        output = self.main2(feature)
        #output = self.main3(output)
        if matching == True:
            return feature,output
        else:
            return output #batch_size x 1 x 1 x 1 => batch_size


netD = _netD()
#netD.apply(weights_init)
# if continue_netD != '':
#     netD.load_state_dict(torch.load(continue_netD))
# print(netD)

def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)

def LSE(before_softmax_output):
    # exp = torch.exp(before_softmax_output)
    # sum_exp = torch.sum(exp,1) #right
    # log_sum_exp = torch.log(sum_exp)
    # return log_sum_exp
    vec = before_softmax_output
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    output = max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast),1))
    return output

def test():
    netD.eval()
    test_loss = 0
    correct = 0
    val_len = 10000
    for data, target in testloader:
        data, target = data.cuda(), torch.LongTensor(target).cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = netD(data)
        test_loss += F.cross_entropy(output, target, size_average=False).data[0] # sum up batch loss
        #test_loss += torch.nn.MultiLabelSoftMarginLoss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cuda().sum()
    test_loss /= len(testloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))
    print >> fsave,'\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(testloader.dataset),100. * correct / len(testloader.dataset))

criterionD = nn.CrossEntropyLoss() # binary cross-entropy
criterionG = nn.MSELoss()
input = torch.FloatTensor(batch_size, 3, imageSize, imageSize)
input_label = torch.FloatTensor(batch_size)
noise = torch.FloatTensor(batch_size, nz, 1, 1)
fixed_noise = torch.FloatTensor(batch_size, nz, 1, 1).normal_(0, 1) #normal_(mean=0, std=1, *, generator=None)
label = torch.FloatTensor(batch_size)
real_label = 1
fake_label = 0


netD.cuda()
netG.cuda()
criterionD.cuda()
criterionG.cuda()
input, label = input.cuda(), label.cuda()
noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

fixed_noise = Variable(fixed_noise) # A fixed (mean, variance) noise distribution # just for testing

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(momentum, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(momentum, 0.999))

#dataloader => batchsize, data, target
for epoch in range(1, epochs + 1):
    for i, data in enumerate(dataloader):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z))) => same as BCELoss
        ###########################
        # train with real
        netD.zero_grad()#set unlearned parameters' gradience to zero.
        real_data, real_label = data
        batch_size = real_data.size(0)
        real_data = real_data.cuda()
        input.resize_as_(real_data).copy_(real_data)
        inputv = Variable(input)
        real_labelv = Variable(real_label)
        input_labelv = Variable(input_label)
        label_input = inputv[:100]
        unlabel_input = inputv[100:]
        l_label = real_label[:100]
        l_labelv = Variable(l_label).cuda()
        l_output = netD(label_input)
        loss_label = criterionD(l_output, l_labelv)
        unl_output = netD(unlabel_input)
        loss_unl_real = -torch.mean(LSE(unl_output),0) +  torch.mean(F.softplus(LSE(unl_output),1),0)
        #train with fake
        noise.resize_(batch_size/2, nz, 1, 1).normal_(0, 1)
        noisev = Variable(noise)
        fake = netG(noisev)
        #labelv = Variable(label.fill_(fake_label))
        unl_output = netD(fake.detach()) #fake images are separated from the graph #results will never gradient(be updated), so G will not be updated
        loss_unl_fake = torch.mean(F.softplus(LSE(unl_output),1),0)
        loss_D = loss_label + loss_unl_real + loss_unl_fake
        loss_D.backward()# because detach(), backward() will not influence netG
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z))) => same as BCELoss
        ###########################
        netG.zero_grad()
        #labelv = Variable(label.fill_(real_label))  # fake labels are real for generator cost
        feature_real,_ = netD(inputv.detach(),matching=True)
        feature_fake,output = netD(fake,matching=True)
        feature_real = torch.mean(feature_real,0)
        feature_fake = torch.mean(feature_fake,0)
        loss_G = criterionG(feature_fake, feature_real.detach())
        loss_G.backward()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_label: %.4f Loss_unlabel_real: %.4f Loss_fake: %.4f Loss_D: %.4f Loss_G: %.4f'
              % (epoch, epochs, i, len(dataloader),loss_label.data[0], loss_unl_real.data[0], loss_unl_fake.data[0], loss_D.data[0], loss_G.data[0]))
        if i % log_interval == 0:
            vutils.save_image(real_data,'%s/real_samples.png' % outf,normalize=True)
            fake = netG(fixed_noise) #just for test
            vutils.save_image(fake.data,'%s/fake_samples_epoch_%03d.png' % (outf, epoch), normalize=True) # batch_size grid
            # .data => transfer Variable() => matrix
        if i % test_interval == 0:
            test()

    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (outf, epoch))

fsave.close()