# improved_gan_pytorch
Pytorch implementation of semi-supervised DCGAN based on "[Improved Techniques for Training GANs](http://arxiv.org/abs/1606.03498)".

Feature matching and semi-supervised GAN have be reimplemented. 

So far, other improved techniques haven't been added. 

# Prerequisites
Pytorch Version: 2.0.3 and Python 2.7

# Usage
Run file： python improved_GAN.py

BTW, in my example, my classifer is for CIFAR10 dataset,

and labeled input : unlabeled input : generated fake input = 1 : 1 : 1

Users also can change the settings according to my program's comments.

P.S. 

For Generator Loss, it is also equal to -loss_unlabled_fake + loss_feature_matching.

For Labeled Loss, it is also equal to -loss_target + log_sum_exp(before_softmax_labeled_output)

# To do
1. to average input labeled data over 10 classes subset.
2. to adjust the network structure for high accuracy classification
3. to reimplement other techniques in improved GAN
4. to reimplement "[Bad GAN](https://arxiv.org/abs/1705.09783)" paper

# Semi-supervised + Feature matching CIFAR10 Classification

![image](https://github.com/eli5168/improved_gan_pytorch/blob/master/example.png)

![image](https://github.com/eli5168/improved_gan_pytorch/blob/master/fake_samples_epoch_300.png)

300th epoch
