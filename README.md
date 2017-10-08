# improved_gan_pytorch
Pytorch implementation of DCGAN based on "[Improved Techniques for Training GANs](http://arxiv.org/abs/1606.03498)".

Feature matching and semi-supervised GAN have be reimplemented. 

So far, other improved techniques haven't been added. 

# Pytorch Version: 2.0.3 and Python 2.7

In my example, my classifer is for CIFAR10 dataset,

and labeled input: unlabeled input : generated fake input = 1 : 1 : 1

Users can change the setting according to my program's comment.

Run file： python improved_GAN.py

P.S. 

For Generator Loss, it is also equal to -loss_unlabled_fake + loss_feature_matching.

For Labeled Loss, it is also equal to -loss_target + log_sum_exp(before_softmax_labeled_output)

# Semi-supervised + Feature matching CIFAR10 Classification

![image](https://github.com/eli5168/improved_gan_pytorch/blob/master/example.png)

![image](https://github.com/eli5168/improved_gan_pytorch/blob/master/fake_samples_epoch_300.png)

300th epoch
