# -*- coding: utf-8 -*- 
import os,sys
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
#import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn.functional as F

fsave = open('accuracy.txt','w')
unlabeled_weight = 1.0 # 
batch_size = 200 #training batch_size
test_batch_size = 64
label_slice = 100
momentum = 0.5 #adam parameter
niter = 100 # number of epochs (training)
lr = 0.0001 #learning rate
imageSize = 32 # resize input image to XX
nz = 100 #size of the latent z vector
ngf = 32 #number of G output filters
ndf = 32 #number of D output filters
nc = 3 # numbel of channel
outf = './fake' #folder to output images and model checkpoints
cudnn.benchmark = True

dataloader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data', train=True, download=False,
                   transform=transforms.Compose([
                   	transforms.ToTensor(),
                   	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   	]))
    ,batch_size=batch_size, shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data', train=False, download=False,
                   transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ]))
    ,batch_size=test_batch_size, shuffle=False, num_workers=2)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class _netG(nn.Module):
    def __init__(self):
        super(_netG, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 2, 1, bias=False), # h = (1-1) * 4 - 0 + 4 + 0 = 4 , w = 0 * 1 - 2 * 0 + 4 + 0 = 4
            nn.BatchNorm2d(ngf * 8),
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
        )

    def forward(self, input):
        output = self.main(input)
        return output


netG = _netG()
netG.apply(weights_init)



class _netD(nn.Module):
    def __init__(self):
        super(_netD, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf ),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            #nn.LeakyReLU(0.2, inplace=True),
        )
        self.main2 = nn.Sequential(
            #nn.Linear(8192,1024),
            #nn.Dropout(0.5),
            nn.Linear(1024,10)
        )
        self.main3 = nn.Sequential(
            nn.Softmax(),
        )
    def forward(self, input, match = False):
        feature = self.main(input)
        #feature = feature.view(-1,8192)
        feature = feature.view(-1,1024)
        before_softmax = self.main2(feature)
        _output = self.main3(before_softmax)
        if match == False:
            return before_softmax,_output
        else:
            return before_softmax, feature, _output


netD = _netD()
netD.apply(weights_init)

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

def lossy(before_softmax_output,label):
    output = torch.FloatTensor(label_slice,1).cuda()
    output = Variable(output)
    for i in range(label_slice):
        output[i] = before_softmax_output[i][label[i]:label[i]+1]
    return output


criterion_D = nn.CrossEntropyLoss() # binary cross-entropy
criterion_G = nn.MSELoss()
#loss_label = torch.Variable()
#loss_unlabel_true = torch.Variable(torch.Tensor(0))
#loss_unlabel_fake = torch.Variable(torch.Tensor(0))

input = torch.FloatTensor(batch_size, 3, imageSize, imageSize)
noise = torch.FloatTensor(batch_size, nz, 1, 1)
fixed_noise = torch.FloatTensor(batch_size, nz, 1, 1).normal_(0, 1) #normal_(mean=0, std=1, *, generator=None)
label = torch.FloatTensor(batch_size)
# fake_label = 10
# true_label = 100

netD.cuda()
netG.cuda()
criterion_D.cuda()
criterion_G.cuda()
#label_input, unlabel_input,label_label = label_input.cuda(), unlabel_input.cuda(), label_label.cuda()
noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

fixed_noise = Variable(fixed_noise) # A fixed (mean, variance) noise distribution # just for testing


# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(momentum, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(momentum, 0.999))



#dataloader => batchsize, data, target
for epoch in range(niter):
    for i, data in enumerate(dataloader):
        label_slice = 100
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z))) => same as BCELoss
        ###########################
        # train with real
        netD.zero_grad()#set unlearned parameters' gradience to zero.
        real_data, real_label = data
        #print batch_size
        try:
            label_data = real_data[:label_slice]
            unlabel_data = real_data[label_slice:]
        except:
            continue
        label_input = torch.FloatTensor(label_slice, nc, imageSize, imageSize)
        unlabel_input = torch.FloatTensor(batch_size-label_slice, nc, imageSize, imageSize)
        label_label = torch.FloatTensor(label_slice)
        label_input, unlabel_input,label_label = label_input.cuda(), unlabel_input.cuda(), label_label.cuda()
        #label_data, unlabel_data = torch.split(real_data,2)
        label_label = real_label[:label_slice]
        # real_data is equal to (batchsize, data) except label
        batch_size = real_data.size(0)
        #label_batch_size = label_data.size(0)
        real_data = real_data.cuda()
        label_data = label_data.cuda()
        unlabel_data = unlabel_data.cuda()
        #input.resize_as_(real_data).copy_(real_data)
        #label.resize_(batch_size).fill_(true_label) 
        label_input.resize_as_(label_data).copy_(label_data)
        unlabel_input.resize_as_(unlabel_data).copy_(unlabel_data)
        #resize_as_ : Resizes the current tensor to be the same size as the specified tensor. This is equivalent to: self.resize_(tensor.size())
        #label.resize_(batch_size).fill_(real_label) # batchsize x fake_
        # Resizes this tensor to the specified size.
        #inputv = Variable(input)
        labelv = Variable(label).cuda()
        label_labelv = Variable(label_label).cuda() #labeled true data's label
        label_inputv = Variable(label_input).cuda()
        unlabel_inputv = Variable(unlabel_input).cuda()
        before_softmax_label, l_output = netD(label_inputv)
        before_softmax_unlabel, u_output = netD(unlabel_inputv)
        loss_label_y = lossy(before_softmax_label,label_labelv.data)
        loss_label_LSE = LSE(before_softmax_label)
        #Loss_label = -torch.mean(loss_label_y,0) + torch.mean(loss_label_LSE,0)
        Loss_label = F.cross_entropy(before_softmax_label,label_labelv)
        Loss_unlabel_real = -torch.mean(LSE(before_softmax_unlabel),0) +  torch.mean(torch.nn.functional.softplus(LSE(before_softmax_unlabel),1),0)

        #train with fake
        noise.resize_(unlabel_data.size(0), nz, 1, 1).normal_(0, 1)
        noisev = Variable(noise)
        fake = netG(noisev)
        before_softmax_fake,_output = netD(fake.detach())#fake images are separated from the graph #results will never gradient(be updated), so G will not be updated
        Loss_unlabel_fake =  torch.mean(torch.nn.functional.softplus(LSE(before_softmax_fake),1),0)
        Loss_D = Loss_label + Loss_unlabel_real + Loss_unlabel_fake
        Loss_D.backward()
        optimizerD.step()
        ############################
        # (2) Update G network: maximize log(D(G(z))) => same as BCELoss
        ###########################
        netG.zero_grad()
        #labelv = Variable(label.fill_(real_label))  # fake labels are real for generator cost
        before_softmax_fake, feature_fake, _output = netD(fake, match = True) 
        #last D has already updated, so needs a new feature_true
        _ , feature_real, _ = netD(unlabel_inputv.detach(), match = True)
        #errG = criterion_G(feature_fake, feature_true) # feature_true do not need to update parameters !!!
        feature_real = torch.mean(feature_real,0)
        feature_fake = torch.mean(feature_fake,0)
        #Loss_G = criterion_G(feature_fake,feature_real)
        Loss_G = -torch.mean(torch.nn.functional.softplus(LSE(before_softmax_fake),1),0)
        #Loss_G += torch.mean(torch.abs(feature_fake-feature_real),0)
        Loss_G += criterion_G(feature_fake,feature_real.detach())
        #errG = criterion_G(output, labelv)
        Loss_G.backward()
        #D_G_z2 = output.data.mean()
        print('[%d/%d][%d/%d] Loss_label: %.4f Loss_unlabel_real: %.4f Loss_fake: %.4f Loss_D: %.4f Loss_G: %.4f'
              % (epoch, niter, i, len(dataloader),Loss_label.data[0], Loss_unlabel_real.data[0], Loss_unlabel_fake.data[0], Loss_D.data[0], Loss_G.data[0]))
        #print >> fsave,'[%d/%d][%d/%d] Loss_label: %.4f Loss_unlabel_real: %.4f Loss_fake: %.4f Loss_D: %.4f Loss_G: %.4f' (epoch, niter, i, len(dataloader),Loss_label.data[0], Loss_unlabel_real.data[0], Loss_unlabel_fake.data[0], Loss_D.data[0], Loss_G.data[0])
        if i % 100 == 0:
            vutils.save_image(real_data,'%s/real_samples.png' % outf,normalize=True)
            fake = netG(fixed_noise) #just for test
            vutils.save_image(fake.data,'%s/fake_samples_epoch_%03d.png' % (outf, epoch), normalize=True) # batch_size grid

    #classifier testing
    netD.eval()
    test_loss = 0
    correct = 0
    for test_data, target in testloader:
        test_data, target = test_data.cuda(), target.cuda()
        test_data, target = Variable(test_data, volatile=True), Variable(target)
        _, _ ,output = netD(test_data, match = True)
        #test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cuda().sum()
    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))
    print >> fsave,'Test set: Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(testloader.dataset),100. * correct / len(testloader.dataset))
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (outf, epoch))
fsave.close()
