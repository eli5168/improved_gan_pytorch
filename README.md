# improved_gan_pytorch
Pytorch implementation of DCGAN based on "[Improved Techniques for Training GANs](http://arxiv.org/abs/1606.03498)".

Feature matching and semi-supervised GAN have be reimplemented.

# Pytorch Version: 2.0.3

In my example, our classifer is for CIFAR10 dataset,

and labeled input: unlabeled input : generated fake input = 1 : 1 : 1

Users can change the setting according to my program's comment.


# Semi-supervised + Feature matching CIFAR10 Classification

![image](https://github.com/eli5168/improved_gan_pytorch/blob/master/acc.png)

Highest accuracy: 98%
